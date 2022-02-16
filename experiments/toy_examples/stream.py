from typing import Generator, Tuple
import torch
import numpy as np
from copy import deepcopy
from scipy.stats import laplace


def mvn(
    dim: int, mean: float=0., var_1_var_2: Tuple[float,float]=(1.,1.)
) -> Generator:
    var_1, var_2 = var_1_var_2
    vars = [var_1 for _ in range(dim//2)] + [var_2 for _ in range(dim-dim//2)]
    while True:
        yield torch.tensor(mean) + (torch.randn(dim) * torch.tensor(vars).sqrt())


def mvl(
    dim: int, mean: float=0., var_1_var_2: Tuple[float,float]=(1.,1.)
) -> Generator:
    var_1, var_2 = var_1_var_2
    vars = [var_1 for _ in range(dim//2)] + [var_2 for _ in range(dim-dim//2)]
    scales = [np.sqrt(var/2) for var in vars]
    while True:
        yield torch.tensor([float(laplace.rvs(mean, scales[i], 1)) for i in range(dim)])


def square(rotated: bool=False, center_removed: bool=False) -> Generator:
    while True:
        xy = np.random.uniform(size=2)
        if center_removed:
            while np.max(np.abs(xy)) < 1/2:
                xy = np.random.uniform(size=2)
        dirs = -1 + 2*(np.random.uniform(size=2) > 0.5).astype(float)
        xy *= dirs
        if rotated:
            rot_mat = np.array([
                [np.cos(np.pi/4), -np.sin(np.pi/4)],
                [np.sin(np.pi/4), np.cos(np.pi/4)]
            ])
            xy = rot_mat @ xy
        yield torch.tensor(xy)


def load_stream(config: dict) -> Generator:
    config = deepcopy(config)
    name = config.pop('name')
    if name == 'mvn':
        stream = mvn(**config)
    elif name =='mvl':
        stream = mvl(**config)
    elif name == 'square':
        stream = square(**config)
    else:
        raise NotImplementedError(f"No implementation found for: {name}")
    return stream