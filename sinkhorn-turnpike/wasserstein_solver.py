from utils import *
from sinkhorn import sinkhorn
from geomloss import SamplesLoss
from geomloss.kernel_samples import gaussian_kernel

class TurnpikeSolver(nn.Module):
    def __init__(self, n_points, lr=1e-4, num_sample=1):
        super(TurnpikeSolver, self).__init__()
        self.n_points = n_points
        self.n_dist = int(n_points * (n_points - 1) / 2)
        self.lr = lr
        self.num_sample = num_sample
        self.params = nn.ParameterDict({
            'x': nn.Parameter(torch.rand(self.n_points, 1, dtype=torch.float64), requires_grad=True)
        })
        self.loss = SamplesLoss(loss="hausdorff", kernel=gaussian_kernel)

    def estimate(self, eps=0., sort=False):
        noise = eps * torch.randn(self.params['x'].shape, dtype=torch.float64).to(device)
        x_est = self.params['x'] + noise
        x_est_rev = 1.0 - x_est
        d_est = points_to_dvec(x_est).view(-1, 1)
        if sort:
            x_est = torch.sort(x_est, dim=0)[0]
            x_est_rev = torch.sort(x_est_rev, dim=0)[0]
            d_est = torch.sort(d_est, dim=0)[0]
        return x_est, x_est_rev, d_est

    def evaluate(self, x, d, tol):
        with torch.no_grad():
            x_est, x_est_rev, d_est = self.estimate(sort=True)
            r = torch.abs(x_est.flatten() - x.flatten())
            r_rev = torch.abs(x_est_rev.flatten() - x.flatten())
            x_bin_acc = max(torch.sum(r < tol).item(), torch.sum(r_rev < tol).item()) / x.shape[0]
            d_bin_acc = torch.sum(torch.abs(d_est.flatten() - d.flatten()) < tol) / d.shape[0]
        return x_bin_acc, d_bin_acc

    def normalize_x(self, x):
        x = x - x.min()
        x = x / x.max()
        return torch.sort(x, dim=0)[0]

    def normalize_d(self, d):
        return d / d.max(), d.max()

    def solve(self, x, d, tol=1e-3, iter=100000, eval_interval=10):
        x, d = x.to(device), d.view(-1, 1).to(device)
        x = self.normalize_x(x)
        d, self.dmax = self.normalize_d(d)
        opt = AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=.5, patience=5)
        bar = trange(iter)
        for i in bar:
            if i % eval_interval == 0:
                x_bin_acc, d_bin_acc = self.evaluate(x, d, tol=tol)
            pmat = torch.cdist(self.params['x'], d)
            pidx = torch.min(pmat, dim=-1).indices
            self.params['x'] = d[pidx]
            opt.zero_grad()
            loss = []
            for j in range(self.num_sample):
                _, _, d_est = self.estimate() #eps = (tol ** 1.5) * (iter - i) / iter)
                loss.append(self.loss(d_est, d))
            loss = torch.stack(loss).mean()
            loss.backward()
            scheduler.step(loss)
            opt.step()
            bar.set_postfix_str(f'Loss={loss.item():.8f} x-bin acc={x_bin_acc:.5f} d-bin acc={d_bin_acc:.5f}')
        return self.estimate()