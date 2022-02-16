from dd.base import ChangeDetector
from dd.kernels import GaussianRBF, Kernel
from dd.projections import Proj, NormalLogpdf, KNN, PCA, UAE, TAE
from dd.lsddinc import LSDDInc, LSDDIncSim
from dd.bstat import B_Stat
from dd.batchmmd import BatchMMDSim


def load_detector(args: list, config: dict) -> ChangeDetector:
    name = config.pop('name')
    if name == 'lsddinc':
        detector = LSDDInc(*args, **config)
    elif name == 'lsddincsim':
        detector = LSDDIncSim(*args, **config)
    elif name == 'bstat':
        detector = B_Stat(*args, **config)
    elif name=='batchmmdsim':
        detector = BatchMMDSim(*args, **config)
    else:
        raise NotImplementedError(f"No implementation found for: {name}")
    return detector


def load_kernel(config: dict) -> Kernel:
    name = config.pop('name')
    if name == 'gaussian_rbf':
        kernel = GaussianRBF(**config)
    else:
        raise NotImplementedError(f"No implementation found for: {name}")
    return kernel


def load_proj(config: dict) -> Proj:
    name = config.pop('name')
    if name == 'mvn_logpdf':
        proj = NormalLogpdf(**config)
    elif name == 'knn':
        proj = KNN(**config)
    elif name =='pca':
        proj = PCA(**config)
    elif name=='uae':
        proj = UAE(**config)
    elif name=='tae':
        proj = TAE(**config)
    else:
        raise NotImplementedError(f"No implementation found for: {name}")
    return proj