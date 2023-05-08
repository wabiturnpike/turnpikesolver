import torch

from multiscale_mask_solver import *
from multiscale_solver import *
from glob import glob
import time
seed()

test_files = glob('/home/qhoang/Code/Turnpike/data/*.pts')
noise = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
res = {}
for tf in test_files:
    tfname = tf.split('/')[-1]
    tokens = tfname.split('_')
    n = int(tokens[2].split('frags')[0])
    if n not in [27, 64, 530]:
        continue
    lines = open(tf, 'r').readlines()
    z = torch.tensor([float(l.strip()) for l in lines], dtype=torch.float64).view(-1, 1).to(device)
    z = (z - z.min()) / (z.max() - z.min())
    d = points_to_dvec(z)

    for eps in noise:
        exp_title = f'{tokens[0]}_{tokens[1]}_{n}_{eps}'
        start_time = time.time()
        d_noise = d + eps * torch.randn(d.shape).to(device)
        d_eps = torch.cat([d_noise, torch.zeros(n).to(device)])
        d_eps = torch.sort(d_eps)[0]
        solver = TurnpikeMultiscaleMaskSolver(n_points=n).to(device)
        z_best, _ = solver.shift_points(z, solver.solve(z, d_eps, iter=20))
        runtime = time.time() - start_time
        res[exp_title] = {'time': runtime, 'est': z_best, 'd_noise': d_noise}
        torch.save(res, './pdtests_estimates/res3.pt')