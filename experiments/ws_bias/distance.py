from typing import List, Callable, Optional, Union, Tuple
import torch
import numpy as np
from dd.utils import zero_diag


def permed_mmds(
    k_ref: torch.Tensor, 
    x_inds: List[torch.tensor], 
    y_inds: List[torch.Tensor]
) -> torch.Tensor:

    n_x = len(x_inds[0])
    n_y = len(y_inds[0])
    assert n_x + n_y == k_ref.shape[0]

    mmds = []
    for x_ind, y_ind in zip(x_inds, y_inds):
        mmd = (
            zero_diag(k_ref[x_ind][:, x_ind]).sum()/(n_x*(n_x-1)) +
            zero_diag(k_ref[y_ind][:, y_ind]).sum()/(n_y*(n_y-1)) -
            2*(k_ref[x_ind][:, y_ind].sum())/(n_x*n_y)
        )
        mmds.append(float(mmd.cpu()))

    return np.array(mmds)


def new_mmds(
    kernel: Callable,
    ref: torch.Tensor,
    k_ref: torch.Tensor,
    x_inds: List[torch.tensor],
    d: int
) -> torch.Tensor:

    n_x = len(x_inds[0])
    n_y = k_ref.shape[0] - n_x

    mmds = []
    for x_ind in x_inds:
        y = torch.randn(n_y, d).to(ref.device)
        k_xy = kernel(ref[x_ind], y)
        k_yy = kernel(y, y)
        mmd = (
            zero_diag(k_ref[x_ind][:, x_ind]).sum()/(n_x*(n_x-1)) +
            zero_diag(k_yy).sum()/(n_y*(n_y-1)) -
            2*(k_xy.sum())/(n_x*n_y)
        )
        mmds.append(float(mmd.cpu()))
    
    return np.array(mmds)


def permed_lsdds(
    k_all_c: torch.Tensor,
    x_perms: List[torch.Tensor],
    y_perms: List[torch.Tensor],
    H: torch.Tensor,
    H_lam_inv: Optional[torch.Tensor] = None,
    lam_rd_max: float = 0.2,
    return_unpermed: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute LSDD estimates from kernel matrix across various ref and test window samples

    Parameters
    ----------
    k_all_c
        Kernel matrix of simmilarities between all samples and the kernel centers.
    x_perms
        List of B reference window index vectors
    y_perms
        List of B test window index vectors
    H
        Special (scaled) kernel matrix of simmilarities between kernel centers
    H_lam_inv
        Function of H corresponding to a particular regulariation parameter lambda.
        See Eqn 11 of Bu et al. (2017)
    lam_rd_max
        The maximum relative difference between two estimates of LSDD that the regularization parameter
        lambda is allowed to cause. Defaults to 0.2. Only relavent if H_lam_inv is not supplied.
    return_unpermed
        Whether or not to return value corresponding to unpermed order defined by k_all_c

    Returns
    -------
    Vector of B LSDD estimates for each permutation, H_lam_inv which may have been inferred, and optionally
    the unpermed LSDD estimate.
    """
    # Compute (for each bootstrap) the average distance to each kernel center (Eqn 7)
    k_xc_perms = torch.stack([k_all_c[x_inds].mean(0) for x_inds in x_perms], 0)
    k_yc_perms = torch.stack([k_all_c[y_inds].mean(0) for y_inds in y_perms], 0)
    h_perms = k_xc_perms - k_yc_perms

    if H_lam_inv is None:
        # We perform the initialisation for multiple candidate lambda values and pick the largest
        # one for which the relative difference (RD) between two difference estimates is below lambda_rd_max.
        # See Appendix A
        candidate_lambdas = [1/(4**i) for i in range(10)]  # TODO: More principled selection
        H_plus_lams = torch.stack(
            [H+torch.eye(H.shape[0], device=H.device)*can_lam for can_lam in candidate_lambdas], 0
        )
        H_plus_lam_invs = torch.inverse(H_plus_lams)
        H_plus_lam_invs = H_plus_lam_invs.permute(1, 2, 0)  # put lambdas in final axis

        omegas = torch.einsum('jkl,bk->bjl', H_plus_lam_invs, h_perms)  # (Eqn 8)
        h_omegas = torch.einsum('bj,bjl->bl', h_perms, omegas)
        omega_H_omegas = torch.einsum('bkl,bkl->bl', torch.einsum('bjl,jk->bkl', omegas, H), omegas)
        rds = (1 - (omega_H_omegas/h_omegas)).mean(0)
        lam_index = (rds < lam_rd_max).nonzero()[0]
        lam = candidate_lambdas[lam_index]
        print(f"Using lambda value of {lam:.2g} with RD of {float(rds[lam_index]):.2g}")
        H_plus_lam_inv = H_plus_lam_invs[:, :, lam_index.item()]
        H_lam_inv = 2*H_plus_lam_inv - (H_plus_lam_inv.transpose(0, 1) @ H @ H_plus_lam_inv)  # (below Eqn 11)

    # Now to compute an LSDD estimate for each permutation
    lsdd_perms = (h_perms * (H_lam_inv @ h_perms.transpose(0, 1)).transpose(0, 1)).sum(-1)  # (Eqn 11)

    if return_unpermed:
        n_x = x_perms[0].shape[0]
        h = k_all_c[:n_x].mean(0) - k_all_c[n_x:].mean(0)
        lsdd_unpermed = (h[None, :] * (H_lam_inv @ h[:, None]).transpose(0, 1)).sum()
        return lsdd_perms, H_lam_inv, lsdd_unpermed
    else:
        return lsdd_perms.cpu().numpy(), H_lam_inv


def new_lsdds(
    kernel: Callable,
    H_lam_inv: torch.tensor,
    ref: torch.tensor,
    k_ref_cs: torch.tensor,
    x_inds: List[torch.tensor],
    cs: torch.tensor,
) -> np.array:

    d = ref.shape[-1]
    n_x = len(x_inds[0])
    n_y = k_ref_cs.shape[0] - n_x
    B = len(x_inds)

    k_xcs = torch.stack([k_ref_cs[x_ind].mean(0) for x_ind in x_inds], 0)
    k_ycs = torch.stack([kernel(torch.randn(n_y, d).to(cs.device), cs).mean(0) for _ in range(B)], 0)
    h_perms = k_xcs - k_ycs

    lsdds = (h_perms * (H_lam_inv @ h_perms.transpose(0, 1)).transpose(0, 1)).sum(-1)  # (Eqn 11)

    return lsdds.cpu().numpy()