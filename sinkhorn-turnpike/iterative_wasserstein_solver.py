from utils import *
from sinkhorn import sinkhorn
from geomloss import SamplesLoss
from geomloss.kernel_samples import gaussian_kernel

class TurnpikeIterativeSolver(nn.Module):
    def __init__(self, n_points, lr=1e-3, blur=0.05):
        super(TurnpikeIterativeSolver, self).__init__()
        self.n_points = n_points
        self.params = nn.ParameterDict({
            'x_opt': nn.Parameter(torch.rand(self.n_points, 1, dtype=torch.float64), requires_grad=True),
            'x_frz': nn.Parameter(torch.rand(self.n_points, 1, dtype=torch.float64), requires_grad=False)
        })
        self.loss = SamplesLoss(loss="hausdorff", kernel=gaussian_kernel, blur=blur)
        self.lr = lr

    def solve(self, x, d, iter=100):
        d, x = d.view(-1,1).to(device), x.to(device)
        opt = AdamW(self.parameters(), lr=self.lr)
        bar = trange(iter)
        max_patience, best_loss = 10, 100.0
        for i in bar:
            r0, decay = 0.8, (iter-i) / iter
            mask = torch.rand(self.n_points, 1) < r0 * (decay ** 2)
            mask = mask.double().to(device)
            pmat = torch.cdist(self.params['x_opt'], d)
            pidx = torch.min(pmat, dim=-1).indices
            self.params['x_frz'] = d[pidx]
            patience, prev_loss = 0, 100.0
            while patience < max_patience:
                opt.zero_grad()
                x_est = mask * self.params['x_frz'] + (1.0 - mask) * self.params['x_opt']
                d_est = points_to_dvec(x_est, diag=0).view(-1, 1)
                loss = self.loss(d_est, d)
                loss.backward()
                opt.step()
                bar.set_postfix_str(f'Loss={loss.item():.8f} Patience={patience}')
                if prev_loss - loss.item() < 1e-7:
                    patience += 1
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    x_best = x_est
                    if best_loss < 1e-8:
                        return x_best
                torch.clamp(self.params['x_opt'],0, 1)
                prev_loss = loss.item()
        return x_best