from abc import ABC, abstractmethod
from typing import Optional
import torch


class ChangeDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, x: torch.tensor) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _process_initial_data(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _initialise(self) -> None:
        raise NotImplementedError()


class BatchCD(ChangeDetector):
    def __init__(
        self,
        initial_data: torch.tensor,
        window_size: int,
        ert: float=1000,
        test_every_k: Optional[int]=None,
    ) -> None:
        super().__init__()
        self.initial_data = initial_data
        self.N = initial_data.shape[0]
        self.window_size = window_size
        self.ert = ert
        self.test_every_k = test_every_k or window_size
        self.fpr = test_every_k/ert
