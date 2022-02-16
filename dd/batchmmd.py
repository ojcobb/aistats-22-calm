from typing import Optional
import torch
from tqdm import tqdm

from dd.base import BatchCD
from dd.kernels import Kernel
from dd.utils import zero_diag, quantile


class BatchMMDSim(BatchCD):
    """Testing using permutation tests"""
    def __init__(
        self,
        kernel: Kernel,
        initial_data: torch.tensor,
        ert: int,
        window_size: int=100,
        test_every_k: Optional[int]=None, # TODO: Not supported
        n_bootstraps: Optional[int]=None
    ):  
        test_every_k = test_every_k or window_size
        super().__init__(initial_data, window_size, ert, test_every_k)
        self.kernel = kernel
        self.n_bootstraps = n_bootstraps or int(10*self.ert/((1-self.fpr)**self.window_size))
        if ert/window_size > self.n_bootstraps:
            raise ValueError("Not possible to achieve an ERT lower than n_bootstraps*window_size") 
        self._process_initial_data()
        self._initialise()
        
    def _initialise(self):
        self.detected = False
        self.t = 0
        self.current_window = []
        self.k_xys = []

        self.ref_inds = torch.randperm(self.N)[:self.rw_size]
        self.k_xx_sum = zero_diag(
            self.kernel_mat[self.ref_inds][:,self.ref_inds]
        ).sum()/(len(self.ref_inds)*(len(self.ref_inds)-1))
        
    
    def _process_initial_data(self) -> None:
        kernel_mat = self.kernel(self.initial_data, configure=True)

        self.rw_size = self.N - (2*self.window_size-1)

        perms = [torch.randperm(self.N) for _ in range(self.n_bootstraps)]
        p_inds_all = [perm[:self.rw_size] for perm in perms]
        q_inds_all = [perm[self.rw_size:] for perm in perms]

        self.thresholds = []
        
        print("Generating permutations of kernel matrix..")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kernel_mat = kernel_mat.to(device)
        k_xy_col_sums_all = [(
            kernel_mat[p_inds][:,q_inds].sum(0)) for p_inds, q_inds in tqdm(zip(p_inds_all, q_inds_all), total=self.n_bootstraps)]
        k_full_sum = zero_diag(kernel_mat).sum()
        k_xx_sums_all = [(
            k_full_sum - zero_diag(kernel_mat[q_inds][:,q_inds]).sum() - 2*k_xy_col_sums.sum()
        )/(self.rw_size*(self.rw_size-1)) for q_inds, k_xy_col_sums in zip(q_inds_all, k_xy_col_sums_all)] # This is bottleneck w.r.t. large n_bootstraps
        k_xy_col_sums_all = [k_xy_col_sums/(self.rw_size*self.window_size) for k_xy_col_sums in k_xy_col_sums_all]

        for w in tqdm(range(self.window_size), "Computing thresholds"):
            q_inds_all_w = [q_inds[w:w+self.window_size] for q_inds in q_inds_all]
            mmds = [(
                k_xx_sum + 
                zero_diag(kernel_mat[q_inds_w][:, q_inds_w]).sum()/(self.window_size*(self.window_size-1)) -
                2*k_xy_col_sums[w:w+self.window_size].sum()
            ) for k_xx_sum, q_inds_w, k_xy_col_sums in zip(k_xx_sums_all, q_inds_all_w, k_xy_col_sums_all)]

            mmds = torch.tensor(mmds)
            
            self.thresholds.append(quantile(mmds, 1-self.fpr))
            q_inds_all = [q_inds_all[i] for i in range(len(q_inds_all)) if mmds[i]<self.thresholds[-1]]
            k_xx_sums_all = [k_xx_sums_all[i] for i in range(len(k_xx_sums_all)) if mmds[i]<self.thresholds[-1]]
            k_xy_col_sums_all = [k_xy_col_sums_all[i] for i in range(len(k_xy_col_sums_all)) if mmds[i]<self.thresholds[-1]]

        self.kernel_mat = kernel_mat.to('cpu')


    def reset(self):
        self._initialise()


    def update(self, x: torch.tensor) -> bool:
        self.t +=1

        self.current_window.append(x)
        self.k_xys.append(self.kernel(self.initial_data[self.ref_inds], x[None,:]))

        if (len(self.current_window)< self.window_size) or (self.t % self.test_every_k != 0):
            return self.detected
        else:
            self.current_window = self.current_window[-self.window_size:]
            self.k_xys = self.k_xys[-self.window_size:]

            cur_win = torch.stack(self.current_window, axis=0)
            k_xy = torch.stack(self.k_xys, axis=0)
            k_yy = self.kernel(cur_win)

            mmd = (
                self.k_xx_sum +
                zero_diag(k_yy).sum()/(self.window_size*(self.window_size-1)) - 
                2*k_xy.mean()
            )
          
            threshold_ind = min(self.t - self.window_size, self.window_size-1)
            if mmd > self.thresholds[threshold_ind]:
                self.detected = True

            return self.detected


def mmd_from_kernel_mat(
    kernel_mat: torch.tensor, m: int, permute: bool=True
) -> torch.tensor:

    kernel_mat = zero_diag(kernel_mat)
    n = kernel_mat.shape[0] - m

    if permute:
        shuffled_inds = torch.randperm(n+m)
        kernel_mat = kernel_mat[shuffled_inds][:,shuffled_inds]
    
    k_yy = kernel_mat[-m:, -m:]
    k_xx = kernel_mat[:-m, :-m]
    k_xy = kernel_mat[-m:, :-m]

    mmd = k_yy.sum()/(m*(m-1)) + k_xx.sum()/(n*(n-1)) - 2*k_xy.mean()

    return mmd
