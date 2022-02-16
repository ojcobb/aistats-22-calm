from typing import Optional
import torch
import numpy as np
from scipy import optimize
from scipy.stats import norm

from dd.base import BatchCD
from dd.kernels import Kernel
from dd.utils import zero_diag


class B_Stat(BatchCD):
    """
    Implementation of https://arxiv.org/abs/1507.01279
    """
    def __init__(
        self,
        kernel: Kernel,
        initial_data: torch.tensor,
        ert: int,
        window_size: int=100,
        test_every_k: int=1,
        n_samples: Optional[int]=None, # for estimating variance and optionally skew
        skew_correction: bool = True,
    ):
        super().__init__(initial_data, window_size, ert, test_every_k)
        self.kernel = kernel
        self.n_blocks = self.N // self.window_size
        self.n_samples = n_samples or int(10*self.ert/((1-self.fpr)**self.window_size))
        self.skew_correction = skew_correction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._process_initial_data()
        self._compute_threshold()
        self._initialise()


    def _process_initial_data(self) -> None:
        self.initial_data = self.initial_data.to(self.device)
        self.kernel_mat = self.kernel(self.initial_data, configure=True)

        self.blocks = self.initial_data.split(self.window_size)[:self.n_blocks]
        ws = self.window_size
        k_xxs = torch.stack([zero_diag(
                self.kernel_mat[(ws*i):(ws*(i+1))][:,(ws*i):(ws*(i+1))]
            ) for i in range(self.n_blocks)
        ])
        self.k_xx_sums = k_xxs.sum(-1).sum(-1)/(ws*(ws-1))

        self.std = np.sqrt(estimate_variance_from_kernel_mat(
            self.kernel_mat, self.window_size, self.n_blocks, self.n_samples
        ))


    def _initialise(self):
        self.detected = False
        self.t = 0
        self.current_window = []
        self.k_xys = [torch.zeros(self.window_size,0).to(self.device) for i in range(self.n_blocks)]


    def _compute_threshold(self):
        skew = estimate_skewness_from_kernel_mat(
            self.kernel_mat, self.window_size, self.n_blocks, self.n_samples
        ) if self.skew_correction else None
        minimand = lambda thresh: np.abs(
            self._compute_ert_given_threshold(thresh, self.window_size, skew=skew) - self.ert
        )
        self.threshold = optimize.minimize_scalar(minimand, method='bounded', bounds=(0,10)).x


    def _compute_ert_given_threshold(self, threshold: float, window_size: int, skew: Optional[float]=None) -> float:
        "Equation 4.4, as ert = fac_1 / (fac_2*fac_3)"
        if threshold==0:
            raise ValueError()
        b = np.array(threshold)
        ws = np.array(window_size)
        if skew is not None:
            k3 = skew * self.std**(-3)
            theta = k3_and_b_to_theta(k3, b)
            fac_1 = np.exp(theta*b - (theta**2)/2 - (k3*theta**3)/6) / b
        else:
            fac_1 = np.exp(b*b/2) / b
        fac_2 = (2*ws-1) / (np.sqrt(2*np.pi)*ws*(ws-1))
        fac_3 = nu(b * np.sqrt((2*(2*ws-1)) / (ws*(ws-1))))
        return fac_1 / (fac_2 * fac_3)


    def reset(self):
        self._initialise()


    def update(self, x: torch.tensor) -> bool:
        self.t += 1
        x = x.to(self.device)
        ws = self.window_size

        self.current_window.append(x)
        k_xyts = self.kernel(self.initial_data, x[None,:]).split(ws)[:self.n_blocks]
        self.k_xys = torch.stack([
            torch.cat([k_xy[:,-ws+1:], k_xyt], axis=1) for
            k_xy, k_xyt in zip(self.k_xys, k_xyts)          
        ])

        if len(self.current_window) < ws:
            return self.detected
        else:
            if len(self.current_window) > ws:
                del self.current_window[0]
            
            cur_win = torch.stack(self.current_window, axis=0)
            k_yy_sum = zero_diag(self.kernel(cur_win)).sum()/(ws*(ws-1))

            k_xy_sums = self.k_xys.sum(-1).sum(-1)/(ws*ws)

            mmds = self.k_xx_sums +  k_yy_sum - 2*k_xy_sums
            distance = mmds.mean()/ self.std

            if distance > self.threshold:
                self.detected = True
            return self.detected


def estimate_variance_from_kernel_mat(
    kernel_mat: torch.tensor,  window_size: int, n_blocks: int, n_samples: int
) -> float:
    "Equation 3.5"
    N = kernel_mat.shape[0]

    x_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_dash_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    y_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    y_dash_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    h_sample = kernel_mat[x_inds, x_dash_inds] + kernel_mat[y_inds, y_dash_inds] - (
        kernel_mat[x_inds, y_dash_inds] + kernel_mat[x_dash_inds, y_inds] 
    )
    avg_h_squared = h_sample.square().mean()/n_blocks
    
    x_ddash_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_dddash_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    h_sample_2 = kernel_mat[x_ddash_inds, x_dddash_inds] + kernel_mat[y_inds, y_dash_inds] - (
        kernel_mat[x_ddash_inds, y_dash_inds] + kernel_mat[x_dddash_inds, y_inds] 
    )
    scaled_covariance = ((n_blocks-1)/n_blocks)*(
        ((h_sample-h_sample.mean()) * (h_sample_2-h_sample_2.mean())).sum() / (n_samples-1)
    )

    window_size_choose_2 = window_size*(window_size-1)/2
    var = (avg_h_squared + scaled_covariance)/window_size_choose_2

    return float(var.cpu())


def estimate_skewness_from_kernel_mat(
    kernel_mat: torch.tensor,  window_size: int, n_blocks: int, n_samples: int
) -> float:

    N = kernel_mat.shape[0]
    ws = window_size

    x_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_d_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_dd_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_ddd_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_dddd_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    x_ddddd_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    y_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    y_d_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)
    y_dd_inds = torch.multinomial(torch.ones(N), n_samples, replacement=True)

    term_1_a = (compute_h_vec_from_kernel_mat(
        kernel_mat, x_inds, x_d_inds, y_inds, y_d_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_d_inds, x_dd_inds, y_d_inds, y_dd_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_dd_inds, x_inds, y_dd_inds, y_inds
    )).mean()

    term_1_b = (compute_h_vec_from_kernel_mat(
        kernel_mat, x_inds, x_d_inds, y_inds, y_d_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_d_inds, x_dd_inds, y_d_inds, y_dd_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_ddd_inds, x_dddd_inds, y_dd_inds, y_inds
    )).mean()

    term_1_c = (compute_h_vec_from_kernel_mat(
        kernel_mat, x_inds, x_d_inds, y_inds, y_d_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_dd_inds, x_ddd_inds, y_d_inds, y_dd_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_dddd_inds, x_ddddd_inds, y_dd_inds, y_inds
    )).mean()

    term_2_a = (compute_h_vec_from_kernel_mat(
        kernel_mat, x_inds, x_d_inds, y_inds, y_d_inds
    )**3).mean()

    term_2_b = (compute_h_vec_from_kernel_mat(
        kernel_mat, x_inds, x_d_inds, y_inds, y_d_inds
    )**2 * compute_h_vec_from_kernel_mat(
        kernel_mat, x_dd_inds, x_ddd_inds, y_inds, y_d_inds
    )).mean()

    term_2_c = (compute_h_vec_from_kernel_mat(
        kernel_mat, x_inds, x_d_inds, y_inds, y_d_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_dd_inds, x_ddd_inds, y_inds, y_d_inds
    ) * compute_h_vec_from_kernel_mat(
        kernel_mat, x_dddd_inds, x_ddddd_inds, y_inds, y_d_inds
    )).mean()

    term_1 = 8*(ws-2)/((ws*(ws-1))**2) * (
        (1/(n_blocks**2)) * term_1_a + 
        (3*(n_blocks-1)/(n_blocks**2)) * term_1_b + 
        ((n_blocks-1)*(n_blocks-2)/(n_blocks**2)) * term_1_c
    )

    term_2 = 4/((ws*(ws-1))**2) * (
        (1/(n_blocks**2)) * term_2_a + 
        (3*(n_blocks-1)/(n_blocks**2)) * term_2_b + 
        ((n_blocks-1)*(n_blocks-2)/(n_blocks**2)) * term_2_c
    )

    return float((term_1 + term_2).cpu().numpy())


def compute_h_vec_from_kernel_mat(
    kernel_mat: torch.tensor, 
    x_inds: torch.tensor, 
    x_dash_inds: torch.tensor,
    y_inds: torch.tensor, 
    y_dash_inds: torch.tensor
) -> torch.tensor:
    h_vec = kernel_mat[x_inds, x_dash_inds] + kernel_mat[y_inds, y_dash_inds] - (
        kernel_mat[x_inds, y_dash_inds] + kernel_mat[x_dash_inds, y_inds] 
    )
    return h_vec

def nu(mu: float) -> float:
    "Equation 4.2"
    numerator = (2/mu)*(norm.cdf(mu/2) - 0.5)
    denominator = (mu/2)*norm.cdf(mu/2) + norm.pdf(mu/2)
    return numerator/denominator

def k3_and_b_to_theta(k3: float, b: float) -> float:
    """Quadratic formula applied to theta + (k3 * theta^2)/ 2 = b"""
    theta = (np.sqrt(1 + 2*k3*b) - 1)/k3
    return theta