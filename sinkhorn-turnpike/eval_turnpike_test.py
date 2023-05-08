from multiscale_mask_solver import *
from multiscale_solver import *
from get_points import *
import time
seed()

res = torch.load('./points_estimates/set2.pt')
mae, mse = {}, {}
times = {}
for k in res.keys():
    mae[k], mse[k], time = 0, 0, 0
    print(k)
    n = k.split('_')[1]
    if n not in times: times[n] = []
    for i, zi in enumerate(res[k]['est']):
        di = torch.sort(points_to_dvec(zi.view(-1, 1)))[0]
        d_noise = torch.sort(res[k]['d_noise'][i])[0]
        mae[k] += torch.abs(di - d_noise).mean().item() / len(res[k]['est'])
        mse[k] += F.mse_loss(di, d_noise).item() / len(res[k]['est'])
        times[n].append(res[k]['time'][i])
    print(k, mae[k], mse[k])

for k in times.keys():
    tk = torch.tensor(times[k])
    print(k, tk.mean(), torch.std(tk))

# NUmber of points : 100 200 300 400 500
# Point distribution: Cauchy Gaussian Uniform
# Noise level: 1e-4 1e-5 ... 1e-8
# 2 numbers MAE & MSE (+- std)

#