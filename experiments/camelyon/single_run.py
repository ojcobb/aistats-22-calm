from typing import Callable, Generator

from dd.base import ChangeDetector

def compute_runtime(
    detector: ChangeDetector,
    stream: Generator,
    change_point: int=0, # 0 equivalent to no changepoint
    distortion: Callable=lambda x: x,
    verbose = False,
    proj: Callable=lambda x: x
) -> float:
    """
    change_point=0 looks at ERT
    change_point>0 looks at EDD and returns the time elapsed past the change
        point **conditional** on making it to the change point
    """
    t = 0
    while True:
        detector.reset()
        while not detector.detected:
            t += 1
            new_x = next(stream)
            if t >= change_point:
                new_x = distortion(new_x)
            new_x = proj(new_x[None,:])[0]
            detector.update(new_x)
        if t >= change_point:
            delay = t-change_point
            if verbose:
                print(delay)
            return delay
        
