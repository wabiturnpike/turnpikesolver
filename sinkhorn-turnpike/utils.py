import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
import time, math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from munkres import Munkres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pykeops
import pykeops.torch as keops

def seed(np_seed = 11041987, torch_seed = 20051987):
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)

def points_to_dvec(points, diag=-1):
    # points: N x D
    n = points.shape[0]
    dmat = torch.cdist(points, points)
    idx = torch.tril_indices(n, n, diag)
    return dmat[idx[0], idx[1]]

def dvec_to_dmat(n, dvec):
    dmat = torch.zeros(n, n)
    idx_bot = torch.tril_indices(n, n, -1)
    idx_top = torch.tril_indices(n, n, 1)
    dmat[idx_bot[0], idx_bot[1]] = dvec
    dmat[idx_top[0], idx_top[1]] = dvec
    return dmat

def prepare_data(n_points, dim=1):
    points = torch.cat([
        torch.rand(n_points - 2, dim, dtype=torch.float64),
        torch.zeros(1, dim, dtype=torch.float64),
        torch.ones(1, dim, dtype=torch.float64)
    ], dim=0)
    dvec = points_to_dvec(points)
    sorted_dvec, _ = torch.sort(dvec)
    return points, sorted_dvec

def prepare_gaussian_data(n_points, dim=1, scale=1):
    points = scale * torch.randn(n_points, dim)
    points = points / torch.norm(points)
    dvec = points_to_dvec(points)
    sorted_dvec, _ = torch.sort(dvec)
    return points, sorted_dvec