from typing import Generator, Callable, List
import torch


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def compute_fpr_from_ert(ert: float, window_size: int, test_every_k: int) -> float:
    """
    Probability new window causes false-alarm = alpha \\
    Expected time to start of such a window = k / alpha (geometric dist)\\
    ERT = W + k * (1/alpha - 1) \\
    """
    if ert <= window_size:
        raise ValueError("Can't achieve an ERT of less than the window size.")
    return 1/((ert-window_size)/test_every_k + 1)


def quantile(sample: torch.tensor, p: float, types: List[int]=[7], sorted: bool=False) -> torch.tensor:
    """
    See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
    Averages estimates corresponding to each type in list
    """
    N = len(sample)
    if not 1/N <= p <= (N-1)/N:
        raise ValueError(f"The {p}-quantile should not be estimated using only {N} samples.")
    if not sorted:
        sorted_sample = sample.sort().values
    
    quantiles = []
    for type in types:
        if type == 6: # With M = k*ert - 1 this one is exact
            h = (N+1)*p
        elif type == 7:
            h = (N-1)*p + 1
        elif type == 8:
            h = (N+1/3)*p + 1/3
        h_floor = int(h)
        quantile = sorted_sample[h_floor-1]
        if h_floor != h:
            quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])
        quantiles.append(quantile)
    return torch.stack(quantiles).mean()


def zero_diag(mat: torch.tensor) -> torch.tensor:
    return mat - torch.diag(mat.diag())


def merge_streams(
    stream_1: Generator, 
    stream_2: Generator,
    change_point: int,
    prob_stream_2: Callable=lambda s: 1 # where s is time since changepoint
) -> Generator:
    
    t = -1
    for x in stream_1:
        t +=1
        if t < change_point:
            yield x
        else:
            if torch.rand(1) < prob_stream_2(t-change_point):
                yield next(stream_2)
            else:
                yield x
