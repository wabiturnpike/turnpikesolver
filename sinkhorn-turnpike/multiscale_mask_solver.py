from utils import *
from sinkhorn import sinkhorn
from geomloss import SamplesLoss
from geomloss.kernel_samples import gaussian_kernel
from diffsort import DiffSortNet

add_sched = lambda x0, xn, t, n, p: x0 - (t ** p) * (x0 - xn) / ((n - 1) ** p)
mul_sched = lambda x0, t, n, p: x0 * ((1 - t / n) ** p)

class TurnpikeMultiscaleMaskSolver(nn.Module):
    def __init__(self, n_points):
        super(TurnpikeMultiscaleMaskSolver, self).__init__()
        self.n_points = n_points
        self.params = nn.ParameterDict({
            'x_opt': nn.Parameter(torch.rand(self.n_points, 1, dtype=torch.float64), requires_grad=True),
            'x_frz': nn.Parameter(torch.rand(self.n_points, 1, dtype=torch.float64), requires_grad=False)
        })

    def shift_points(self, x, x_best):
        x_best_shift = x_best + torch.mean(x - x_best)
        d_best = points_to_dvec(x_best.view(-1, 1), diag=0)
        d_best = torch.sort(d_best)[0]
        x_best_shift = torch.sort(x_best_shift.flatten())[0]
        return x_best_shift, d_best

    def evaluate(self, x, x_best, d):
        x_best_shift, d_best = self.shift_points(x, x_best)
        x = torch.sort(x.flatten())[0]
        rd = torch.abs(d_best - d.flatten())
        rx = torch.abs(x_best_shift - x)
        tol = [1e-3, 1e-4, 1e-5, 1e-6]
        return [(torch.sum(rd < t).item(), torch.sum(rx < t).item()) for t in tol]

    def solve(self, x, d, iter=20, init_blur=5e-2, target_blur=1e-5, mode='fixed'):
        def _patient_inner_loop(improv_thres=1e-12):
            patience, max_patience, best_loss = 0, 20, 100.0
            x_best = self.params['x_opt']
            while patience < max_patience:
                opt.zero_grad()
                x_est = mask * self.params['x_frz'] + (1.0 - mask) * self.params['x_opt']
                d_est = points_to_dvec(x_est, diag=0).view(-1, 1)
                loss = loss_fn(d_est, d)
                loss.backward()
                opt.step()
                bar.set_postfix_str(f'Loss={loss.item():.15f} Patience={patience} Blur={blur:.4f}')
                if loss.item() < best_loss:
                    patience = 0 if loss.item() + improv_thres < best_loss else patience + 1
                    best_loss = loss.item()
                    x_best = x_est
                else:
                    patience +=1
                torch.clamp(self.params['x_opt'],0, 1)
            return x_best

        def _fixed_inner_loop(inner_loop_itr=100):
            best_loss = 100.
            x_best = self.params['x_opt']
            for t in range(inner_loop_itr):
                opt.zero_grad()
                x_est = mask * self.params['x_frz'] + (1.0 - mask) * self.params['x_opt']
                d_est = points_to_dvec(x_est, diag=0).view(-1, 1)
                loss = loss_fn(d_est, d)
                loss.backward()
                opt.step()
                bar.set_postfix_str(f'Loss={loss.item():.15f} Itr={t} Blur={blur:.4f}')
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    x_best = x_est
                torch.clamp(self.params['x_opt'], 0, 1)
            return x_best

        def _fixed_inner_loop_sym(inner_loop_itr=100):
            best_loss = 100.
            x_best = self.params['x_opt']
            for t in range(inner_loop_itr):
                opt.zero_grad()
                x_est_1 = mask * self.params['x_frz'] + (1.0 - mask) * self.params['x_opt']
                x_est_2 = (1.0 - mask) * self.params['x_frz'] + mask * self.params['x_opt']
                d_est_1 = points_to_dvec(x_est_1, diag=0).view(-1, 1)
                d_est_2 = points_to_dvec(x_est_2, diag=0).view(-1, 1)
                loss_1 = loss_fn(d_est_1, d)
                loss_2 = loss_fn(d_est_2, d)
                loss = torch.min(loss_1, loss_2)
                loss.backward()
                opt.step()
                bar.set_postfix_str(f'Loss={loss.item():.15f} Itr={t} Blur={blur:.4f}')
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    x_best = x_est_1 if loss_1 < loss_2 else x_est_2
                torch.clamp(self.params['x_opt'],0, 1)
            return x_best

        def _select_random_mask(rate):
            return (torch.rand(self.n_points, 1) < rate).double()

        d = d.view(-1,1)
        bar = trange(iter)
        for i in bar:
            blur = add_sched(init_blur, target_blur, i, iter, .3) # if i < 100 else target_blur
            opt = AdamW(self.parameters(), lr= blur / 10)
            loss_fn = SamplesLoss(loss="hausdorff", kernel=gaussian_kernel, blur=blur)
            mask = _select_random_mask(add_sched(.5, .9, i, iter, 1)).to(x.device)
            pmat = torch.cdist(self.params['x_opt'], d)
            pidx = torch.min(pmat, dim=-1).indices
            self.params['x_frz'] = d[pidx]

            if mode == 'patient':
                x_best = _patient_inner_loop(1e-12)
            elif mode == 'fixed':
                x_best = _fixed_inner_loop(20)
            elif mode == 'fixed_sym':
                x_best = _fixed_inner_loop_sym(30)
            bar.set_postfix_str(f'Binning acc: {self.evaluate(x, x_best, d)}')
            print('')
        return x_best