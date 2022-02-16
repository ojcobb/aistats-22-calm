from typing import Optional
from abc import ABC, abstractmethod

import torch


class Kernel(ABC):
    params: dict
    max: float
    @abstractmethod
    def __call__(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError()


class GaussianRBF(Kernel):
    """
    Note convention with variance
    """
    def __init__(self, var: float=1.) -> None:
        self.params = {'var': var}
        self.max = 1.

    def __call__(self, 
        x: torch.tensor, y: Optional[torch.tensor]=None, configure: bool=False
    ) -> torch.tensor:

        y = x if y is None else y
        l2_mat_squared = (x[:,None,:]-y[None,:, :]).square().sum(-1)

        if configure:
            n = x.shape[0]
            sorted_squared_distances = l2_mat_squared.flatten().sort().values
            median_index = torch.tensor(n*(n-1)/2 + n).int() # +n to account for diagonal
            median_squared_dist = sorted_squared_distances[median_index]
            self.params['var'] = median_squared_dist.item()

        var = torch.tensor(self.params['var'])
        kernel_mat = (-1*(l2_mat_squared/(2*var))).exp()

        return kernel_mat