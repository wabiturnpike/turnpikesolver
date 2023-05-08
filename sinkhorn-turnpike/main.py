from wasserstein_solver import *
from iterative_wasserstein_solver import *
from multiscale_mask_solver import *

seed()
n, noise = 100, 1e-5
x = torch.randn(n, 1, dtype=torch.float64)
x = x - x.min()
x = x / x.max()
d = points_to_dvec(x)
d_noise = d + noise * torch.randn(d.shape)
d = torch.cat([d_noise, torch.zeros(n)])
d = torch.sort(d)[0]
solver = TurnpikeIterativeSolver(n_points=n).to(device)
x_best = solver.solve(x, d)

# Polish result
tol = [1e-3, 1e-4, 1e-5, 1e-6]
for t in tol:
    polish_solver = TurnpikeIterativeSolver(n_points=n, blur=tol).to(device)
    polish_solver.params['x_opt'] = x_best
    x_best_t = solver.solve(x, d, iter=1)
    x_best_t_shift = x_best_t - x_best_t.min()
    d_best = points_to_dvec(x_best_t_shift.view(-1, 1))
    d_best = torch.sort(d_best)[0]
    r = torch.abs(d_best - torch.sort(d_noise.to(device))[0])
    print(f'Tol={t} --> Acc={torch.sum(r < t)}')