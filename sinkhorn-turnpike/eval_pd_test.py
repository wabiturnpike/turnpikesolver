from multiscale_mask_solver import *
from multiscale_solver import *
from get_points import *
import time
seed()

res = torch.load('./pdtests_estimates/res.pt')
mae, mse = {}, {}
for k in res.keys():
    noise = float(k.split('_')[-1])
    d = torch.sort(points_to_dvec(res[k]['est'].view(-1, 1)))[0]
    d_noise = torch.sort(res[k]['d_noise'])[0]
    mae[k] = torch.abs(d - d_noise).mean().item()
    mse[k] = F.mse_loss(d, d_noise).item()
    print(f'{k} {mae[k] / noise:.3e} {mse[k] / noise:3e}')