from typing import Optional
import torch
from tqdm import tqdm

from dd.base import BatchCD
from dd.kernels import GaussianRBF
from dd.utils import quantile


class LSDDInc(BatchCD):
    """
    Implementation of Bu et al. 2017:
    'An Incremental Change Detection Test Based on Density Difference Estimation'
    Variable names mirror notation from paper.
    This is NOT a kernel-based method in the same way as others.
    It uses the Gaussian RBF in forming an estimate of LSDD
    """
    def __init__(
        self,
        initial_data: torch.tensor,
        ert: int,
        window_size: int=100,
        test_every_k: int=1,
        lambda_rd_max: float=0.2,
        normalise: bool=False,
        n_bootstraps: Optional[int]=None,
        training_n: Optional[int]=None,
        test_n: Optional[int]=None,
        n_kernel_centers: Optional[int]=None
    ):  
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_data = initial_data.to(self.device)

        if normalise:
            self.og_mean = initial_data.mean(0)
            self.og_std = initial_data.std(0)
            initial_data = (initial_data-self.og_mean)/self.og_std

        self.n_kernel_centers = n_kernel_centers or 2*window_size
        initial_data, self.kernel_centers = initial_data.split(len(initial_data)-self.n_kernel_centers)
        super().__init__(initial_data, window_size, ert, test_every_k)

        kernel = GaussianRBF()
        _ = kernel(initial_data, configure=True)
        self.rbf_var = kernel.params['var']

        self.lambda_rd_max = lambda_rd_max
        self.normalise = normalise
        self.n_bootstraps = n_bootstraps or int(10*self.ert/((1-self.fpr)**self.window_size))
        self.training_n = training_n
        self.test_n = test_n
        
        self._process_initial_data()
        self._initialise()

    def _process_initial_data(self) -> None:

        training_n = self.training_n or self.window_size

        kernel_center_inds = torch.multinomial(
            torch.ones(self.N), training_n+self.window_size, replacement=False
        )
        kernel_centers = self.initial_data[kernel_center_inds]
        k_data_centers = GaussianRBF(self.rbf_var)(self.initial_data, kernel_centers)

        p_inds = torch.multinomial(torch.ones(self.N), self.n_bootstraps*training_n, replacement=True)
        q_inds = torch.multinomial(torch.ones(self.N), self.n_bootstraps*self.window_size, replacement=True)
        p_ks = k_data_centers[p_inds].reshape(self.n_bootstraps, training_n, -1)
        q_ks = k_data_centers[q_inds].reshape(self.n_bootstraps, self.window_size, -1)
        hs = p_ks.mean(1) - q_ks.mean(1)

        eps=1e-8
        candidate_lambdas = [1/(2**i)-eps for i in range(10)]
        H = GaussianRBF(2*self.rbf_var)(kernel_centers)
        H_plus_lams = torch.stack([H+torch.eye(H.shape[0], device=self.device)*can_lam for can_lam in candidate_lambdas], axis=0)
        H_plus_lam_invs = torch.inverse(H_plus_lams).permute(1,2,0) # lambdas last

        omegas = torch.einsum('jkl,bk->bjl', H_plus_lam_invs, hs)
        h_omegas = torch.einsum('bj,bjl->bl', hs, omegas)
        omega_H_omegas = torch.einsum('bkl,bkl->bl', torch.einsum('bjl,jk->bkl', omegas, H), omegas)
        
        rds = (1 - (omega_H_omegas/h_omegas)).mean(0)
        lambda_index = (rds<self.lambda_rd_max).nonzero()[0]
        lam = candidate_lambdas[lambda_index]
        print(f"Using lambda value of {lam:.2g} with RD of {rds[lambda_index].item():.2g}")

        H_plus_lam_inv = H_plus_lam_invs[:,:,lambda_index.item()]
        self.H_lam_inv = 2*H_plus_lam_inv - (H_plus_lam_inv.transpose(0,1) @ H @ H_plus_lam_inv)

        # distances = (2*h_omegas[:,lambda_index] - omega_H_omegas[:, lambda_index])[:,0]
        distances = (hs * (self.H_lam_inv @ hs.transpose(0,1)).transpose(0,1)).sum(-1) # same thing
        threshold_n = quantile(torch.tensor(distances), 1-self.fpr)

        self.test_n = self.test_n or self.N
        if self.test_n != training_n: # Eqn 19
            self.threshold = ((1/self.test_n + 1/self.window_size)/(1/training_n + 1/self.window_size)-1)*(
                distances.mean()) + threshold_n
        else:
            self.threshold = threshold_n

        self.c2s = k_data_centers.mean(0)
        self.kernel_centers = kernel_centers
        self.k_data_centers = k_data_centers
    
    def _initialise(self):
        self.detected = False
        self.t = 0
        self.current_window = []
        self.k_window_centers = []

    def reset(self):
        self._initialise()

    def update(self, x: torch.tensor) -> bool:
        self.t += 1
        x = x.to(self.device)
        
        if self.normalise:
            x = (x-self.og_mean)/self.og_std

        self.current_window.append(x)
        self.k_window_centers.append(GaussianRBF(self.rbf_var)(x[None,:], self.kernel_centers))

        if len(self.current_window) < self.window_size:
            return self.detected
        else:
            if len(self.current_window) > self.window_size:
                del self.current_window[0]
                del self.k_window_centers[0]

            if self.test_n != self.N:
                ref_inds = torch.multinomial(torch.ones(self.N), self.test_n, replacement=True)
                self.c2s = self.k_data_centers[ref_inds].mean(0)

            hs = self.c2s - torch.cat(self.k_window_centers, axis=0).mean(0)
            distance = hs[None,:] @ self.H_lam_inv @ hs[:, None]

            if distance > self.threshold:
                self.detected = True
            return self.detected


class LSDDIncSim(BatchCD):
    """
    Adapted implementation of Bu et al. 2017:
    'An Incremental Change Detection Test Based on Density Difference Estimation'
    Variable names mirror notation from paper.
    This is NOT a kernel-based method in the same way as others.
    It uses the Gaussian RBF in forming an estimate of LSDD
    """
    def __init__(
        self,
        initial_data: torch.tensor,
        ert: int,
        window_size: int=100,
        test_every_k: int=1,
        lambda_rd_max: float=0.2,
        normalise: bool=False,
        n_bootstraps: Optional[int]=None,
        n_kernel_centers: Optional[int]=None
    ):  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial_data = initial_data.to(self.device)

        if normalise:
            self.og_mean = initial_data.mean(0)
            self.og_std = initial_data.std(0)
            initial_data = (initial_data-self.og_mean)/self.og_std

        self.n_kernel_centers = n_kernel_centers or 2*window_size
        initial_data, self.kernel_centers = initial_data.split(len(initial_data)-self.n_kernel_centers)
        super().__init__(initial_data, window_size, ert, test_every_k)

        kernel = GaussianRBF()
        _ = kernel(initial_data, configure=True)
        self.rbf_var = kernel.params['var']

        self.lambda_rd_max = lambda_rd_max
        self.normalise = normalise
        self.n_bootstraps = n_bootstraps or int(10*self.ert/((1-self.fpr)**self.window_size)
        )
        self._process_initial_data()
        self._initialise()


    def _process_initial_data(self) -> None:

        etw_size = 2*self.window_size - 1
        rw_size = self.N - etw_size
            
        k_data_centers = GaussianRBF(self.rbf_var)(self.initial_data, self.kernel_centers)

        perms = [torch.randperm(self.N) for _ in range(self.n_bootstraps)]
        p_inds_all = [perm[:rw_size] for perm in perms]
        q_inds_all = [perm[rw_size:] for perm in perms]
    
        p_ks = torch.stack([k_data_centers[p_inds].mean(0) for p_inds in p_inds_all], axis=0)
        q_ks = torch.stack([k_data_centers[q_inds[:self.window_size]].mean(0) for q_inds in q_inds_all], axis=0)
        hs = p_ks - q_ks

        eps=1e-8
        candidate_lambdas = [1/(2**i)-eps for i in range(10)]
        H = GaussianRBF(2*self.rbf_var)(self.kernel_centers)
        H_plus_lams = torch.stack([H+torch.eye(H.shape[0], device=self.device)*can_lam for can_lam in candidate_lambdas], axis=0)
        H_plus_lam_invs = torch.inverse(H_plus_lams).permute(1,2,0) # lambdas last

        omegas = torch.einsum('jkl,bk->bjl', H_plus_lam_invs, hs)
        h_omegas = torch.einsum('bj,bjl->bl', hs, omegas)
        omega_H_omegas = torch.einsum('bkl,bkl->bl', torch.einsum('bjl,jk->bkl', omegas, H), omegas)
        
        rds = (1 - (omega_H_omegas/h_omegas)).mean(0)
        lambda_index = (rds<self.lambda_rd_max).nonzero()[0]
        lam = candidate_lambdas[lambda_index]
        print(f"Using lambda value of {lam:.4g} with RD of {rds[lambda_index].item():.2g}")

        H_plus_lam_inv = H_plus_lam_invs[:,:,lambda_index.item()]
        self.H_lam_inv = 2*H_plus_lam_inv - (H_plus_lam_inv.transpose(0,1) @ H @ H_plus_lam_inv)

        # distances = (2*h_omegas[:,lambda_index] - omega_H_omegas[:, lambda_index])[:,0]
        distances = (hs * (self.H_lam_inv @ hs.transpose(0,1)).transpose(0,1)).sum(-1) # same thing

        self.thresholds = [quantile(torch.tensor(distances), 1-self.fpr)]
        p_ks = torch.stack([k_data_centers[p_inds].mean(0) for p_inds in p_inds_all], axis=0)
        for w in tqdm(range(1, self.window_size), "Computing thresholds"):
            p_ks = p_ks[distances<self.thresholds[-1]]
            q_inds_all = [q_inds_all[i] for i in range(len(q_inds_all)) if distances[i]<self.thresholds[-1]]
            q_ks = torch.stack([k_data_centers[q_inds[w:(w+self.window_size)]].mean(0) for q_inds in q_inds_all], axis=0)
            hs = p_ks - q_ks
            distances = (hs * (self.H_lam_inv @ hs.transpose(0,1)).transpose(0,1)).sum(-1)
            self.thresholds.append(quantile(torch.tensor(distances), 1-self.fpr).to('cpu'))
        
        self.k_data_centers = k_data_centers


    def _initialise(self):
        self.detected = False
        self.t = 0
        self.current_window = []
        self.k_window_centers = []

        self.ref_inds = torch.randperm(self.N)[:(-2*self.window_size+1)]
        self.c2s = self.k_data_centers[self.ref_inds].mean(0)


    def reset(self):
        self._initialise()


    def update(self, x: torch.tensor) -> bool:
        self.t += 1
        x = x.to(self.device)

        if self.normalise:
            x = (x-self.og_mean)/self.og_std

        self.current_window.append(x)
        self.k_window_centers.append(GaussianRBF(self.rbf_var)(x[None,:], self.kernel_centers))

        if len(self.current_window) < self.window_size:
            return self.detected
        else:
            if len(self.current_window) > self.window_size:
                del self.current_window[0]
                del self.k_window_centers[0]

            hs = self.c2s - torch.cat(self.k_window_centers, axis=0).mean(0)
            distance = hs[None,:] @ self.H_lam_inv @ hs[:, None]

            threshold_ind = min(self.t - self.window_size, self.window_size-1)

            if distance > self.thresholds[threshold_ind]:
                self.detected = True
            return self.detected