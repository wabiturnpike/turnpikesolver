from multiscale_mask_solver import *
from multiscale_solver import *
from get_points import *
import time
seed()

distribution = [
    'uniform',
    'normal',
    'cauchy'
]
sets = {
    'set1': [n for n in range(100, 401, 100)],
    'set2': [n for n in range(400, 401, 100)],
    'set3': [n for n in range(500, 501, 100)],
    'set4': [n for n in range(400, 401, 100)],
    'set5': [n for n in range(500, 501, 100)]
}
noise = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
res = {}
set = 'set5'
if set == 'set3':
    noise = [1e-4, 1e-5, 1e-6]
if set == 'set4':
    noise = [1e-8]
if set == 'set5':
    noise = [1e-7, 1e-8]
for d in distribution:
    for n in sets[set]:
        for eps in noise:
            exp_title = f'{d}_{n}_{eps}'
            z = torch.tensor(read_points(d, n))
            d_noise = []
            z_est = []
            runtime = []
            for i in range(5):
                start_time = time.time()
                zi = (z[i] - z[i].min()) / (z[i].max() - z[i].min())
                zi = zi.view(-1, 1).to(device)
                di = points_to_dvec(zi)
                di_noise = di + eps * torch.randn(di.shape).to(device)
                di = torch.cat([di_noise, torch.zeros(n).to(device)])
                di = torch.sort(di)[0]
                solver = TurnpikeMultiscaleMaskSolver(n_points=n).to(device)
                z_best, _ = solver.shift_points(zi, solver.solve(zi, di, iter=20))
                z_est.append(z_best.flatten())
                runtime.append(time.time() - start_time)
                d_noise.append(di_noise)
            res[exp_title] = {'time': runtime, 'est': z_est, 'd_noise': d_noise}
            torch.save(res, f'./points_estimates/{set}.pt')