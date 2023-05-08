from utils import *
from sinkhorn import sinkhorn
from geomloss import SamplesLoss
from geomloss.kernel_samples import gaussian_kernel

add_sched = lambda x0, xn, t, n, p: x0 - (t ** p) * (x0 - xn) / ((n - 1) ** p)
mul_sched = lambda x0, t, n, p: x0 * ((1 - t / n) ** p)

class TurnpikeMultiscaleSolver(nn.Module):
    def __init__(self, n_points):
        super(TurnpikeMultiscaleSolver, self).__init__()
        self.n_points = n_points
        self.params = nn.ParameterDict({
            'x_opt': nn.Parameter(torch.rand(self.n_points, 1, dtype=torch.float64), requires_grad=True)
        })

    def evaluate(self, x, x_best, d):
        x_best_shift = x_best + torch.mean(x - x_best)
        d_best = points_to_dvec(x_best.view(-1, 1), diag=0)
        d_best = torch.sort(d_best)[0]
        x_best_shift = torch.sort(x_best_shift.flatten())[0]
        x = torch.sort(x.flatten())[0]
        rd = torch.abs(d_best - d.flatten())
        rx = torch.abs(x_best_shift - x)
        tol = [1e-3, 1e-4, 1e-5, 1e-6]
        return [(torch.sum(rd < t).item(), torch.sum(rx < t).item()) for t in tol]

    def solve(self, x, d, iter=100, init_blur=5e-2, target_blur=1e-4, scale=1):
        d, x = d.view(-1,1).to(device), x.to(device)
        bar = trange(iter)
        max_patience = 10
        for i in bar:
            blur = add_sched(init_blur, target_blur, i, iter, 1)
            opt = AdamW(self.parameters(), lr= blur / 10)
            loss_fn = SamplesLoss(loss="hausdorff", kernel=gaussian_kernel, blur=blur)
            pmat = torch.cdist(self.params['x_opt'], d)
            pidx = torch.min(pmat, dim=-1).indices
            self.params['x_opt'] = d[pidx]
            patience, prev_loss, best_loss = 0, 100.0, 100.0
            while patience < max_patience + int(i ** 0.7):
                opt.zero_grad()
                d_est = points_to_dvec(self.params['x_opt'], diag=0).view(-1, 1)
                loss = loss_fn(d_est * scale, d * scale)
                loss.backward()
                opt.step()
                bar.set_postfix_str(f'Loss={loss.item():.10f} Patience={patience} Blur={blur:.4f}')
                if loss.item() + 1e-10 < best_loss:
                    patience = 0
                    best_loss = loss.item()
                    x_best = self.params['x_opt']
                else:
                    patience +=1
                torch.clamp(self.params['x_opt'],0, 1)
            bar.set_postfix_str(f'Binning acc: {self.evaluate(x, x_best, d)}')
            print('')
        return x_best