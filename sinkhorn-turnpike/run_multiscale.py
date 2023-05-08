from wasserstein_solver import *
from iterative_wasserstein_solver import *
from multiscale_mask_solver import *
from multiscale_solver import *

seed()
n, noise = 100, 1e-5
while 1:
    x = torch.randn(n, 1, dtype=torch.float64)
    x = x - x.min()
    x = x / x.max()
    d = points_to_dvec(x)
    if d.min() >= 1e-4:
        break
d_noise = d + noise * torch.randn(d.shape)
d = torch.cat([d_noise, torch.zeros(n)])
d = torch.sort(d)[0]


solver = TurnpikeMultiscaleMaskSolver(n_points=n).to(device)
x_best = solver.solve(x, d)
solver.evaluate(x.to(device), x_best, d.to(device))