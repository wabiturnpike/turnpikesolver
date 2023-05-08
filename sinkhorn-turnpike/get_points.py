import os
import pickle
import numpy as np


def read_points(dist: str, n: int):
    ns = range(25, 2001, 25)
    dists = ('cauchy', 'normal', 'uniform')
    
    assert n in ns, f"n must be in {ns}"
    assert dist in dists, f"dist must be in {dists}"
    
    save_path = os.path.join('points', f'{dist}_{n}.pb')
    
    with open(save_path, 'rb') as pb:
        Z = pickle.load(pb)
    
    return Z


if __name__ == '__main__':
    pass

