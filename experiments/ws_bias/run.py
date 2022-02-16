import torch
import numpy as np
import os
import argparse
import yaml
from datetime import datetime

from dd.utils import DictAsMember
from dd.kernels import GaussianRBF

from experiments.ws_bias.distance import permed_lsdds, new_lsdds, permed_mmds, new_mmds

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sb


def run(
    exp: int,
    cfgs: dict
):

    cfg = DictAsMember(cfgs[exp])
    stamped_exp = str(exp) + datetime.now().strftime("_%m-%d_%H:%M")
    save_dir = os.path.join('logs', stamped_exp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = torch.randn(cfg.N, cfg.d).to(device)
    kernel = GaussianRBF()
    k_ref = kernel(ref, ref, configure=True)

    # for lsdd
    var = kernel.params['var']
    cs = torch.randn(2*cfg.W, cfg.d).to(device)
    k_ref_cs = kernel(ref, cs) 
    H = GaussianRBF(2*var)(cs, cs)

    # with replacement
    x_inds_r = [torch.multinomial(torch.ones(cfg.N), cfg.N - cfg.W) for _ in range(cfg.B)]
    y_inds_r = [torch.multinomial(torch.ones(cfg.N), cfg.W) for _ in range(cfg.B)]

    lsdds_bs_r, H_lam_inv = permed_lsdds(k_ref_cs, x_inds_r, y_inds_r, H)
    mmds_bs_r = permed_mmds(k_ref, x_inds_r, y_inds_r)

    x_inds_new_r = [torch.multinomial(torch.ones(cfg.N), cfg.N - cfg.W) for _ in range(cfg.B)]
    lsdds_new_r = new_lsdds(kernel, H_lam_inv, ref, k_ref_cs, x_inds_new_r, cs)
    mmds_new_r = new_mmds(kernel, ref, k_ref, x_inds_new_r, cfg.d)

    # without replacement
    perms = [torch.randperm(cfg.N) for _ in range(cfg.B)]
    x_inds_nr = [perm[:-cfg.W] for perm in perms]
    y_inds_nr = [perm[-cfg.W:] for perm in perms]

    mmds_bs_nr = permed_mmds(k_ref, x_inds_nr, y_inds_nr)
    lsdds_bs_nr, H_lam_inv = permed_lsdds(k_ref_cs, x_inds_nr, y_inds_nr, H)

    x_inds_new_nr = [torch.randperm(cfg.N)[:-cfg.W] for _ in range(cfg.B)]
    mmds_new_nr = new_mmds(kernel, ref, k_ref, x_inds_new_nr, cfg.d)
    lsdds_new_nr = new_lsdds(kernel, H_lam_inv, ref, k_ref_cs, x_inds_new_nr, cs)
    
    output_dir = os.path.join('outputs', f"d_{cfg.d}_N_{cfg.N}_W_{cfg.W}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    results = {
        'mmd': {
            'with': {
                'ks': float(ks_2samp(mmds_bs_r, mmds_new_r).statistic),
                '99q': float((mmds_new_r < np.quantile(mmds_bs_r, 0.99)).astype(float).mean()),
                '999q': float((mmds_new_r < np.quantile(mmds_bs_r, 0.999)).astype(float).mean())
            },
            'without': {
                'ks': float(ks_2samp(mmds_bs_nr, mmds_new_nr).statistic),
                '99q': float((mmds_new_nr < np.quantile(mmds_bs_nr, 0.99)).astype(float).mean()),
                '999q': float((mmds_new_nr < np.quantile(mmds_bs_nr, 0.999)).astype(float).mean())
            }
        },
        'lsdd': {
            'with': {
                'ks': float(ks_2samp(lsdds_bs_r, lsdds_new_r).statistic),
                '99q': float((lsdds_new_r < np.quantile(lsdds_bs_r, 0.99)).astype(float).mean()),
                '999q': float((lsdds_new_r < np.quantile(lsdds_bs_r, 0.999)).astype(float).mean())
            },
            'without': {
                'ks': float(ks_2samp(lsdds_bs_nr, lsdds_new_nr).statistic),
                '99q': float((lsdds_new_nr < np.quantile(lsdds_bs_nr, 0.99)).astype(float).mean()),
                '999q': float((lsdds_new_nr < np.quantile(lsdds_bs_nr, 0.999)).astype(float).mean())
            }
        }
    }
    with open(os.path.join(output_dir, 'results.yaml'), 'w+') as fd:
        yaml.dump(results, fd)

    # normalise for plots
    mmds_bs_r, mmds_new_r = mmds_bs_r/np.abs(mmds_bs_r.mean()),  mmds_new_r/np.abs(mmds_bs_r.mean())
    mmds_bs_nr, mmds_new_nr = mmds_bs_nr/np.abs(mmds_bs_nr.mean()),  mmds_new_nr/np.abs(mmds_bs_nr.mean())
    lsdds_bs_r, lsdds_new_r = lsdds_bs_r/np.abs(lsdds_bs_r.mean()),  lsdds_new_r/np.abs(lsdds_bs_r.mean())
    lsdds_bs_nr, lsdds_new_nr = lsdds_bs_nr/np.abs(lsdds_bs_nr.mean()),  lsdds_new_nr/np.abs(lsdds_bs_nr.mean())

    # mmd with
    plt.figure(figsize=(5,5))
    sb.kdeplot(mmds_bs_r, label="with rep")
    sb.kdeplot(mmds_new_r, label="target")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mmd_with"))

    # mmd w/o
    plt.figure(figsize=(5,5))
    sb.kdeplot(mmds_bs_nr, label="without rep")
    sb.kdeplot(mmds_new_nr, label="target")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mmd_without"))
    
    # lsdd with
    plt.figure(figsize=(5,5))
    sb.kdeplot(lsdds_bs_r, label="with rep")
    sb.kdeplot(lsdds_new_r, label="target")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "lsdd_with"))

    # lsdd w/o
    plt.figure(figsize=(5,5))
    sb.kdeplot(lsdds_bs_nr, label="without rep")
    sb.kdeplot(lsdds_new_nr, label="target")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "lsdd_without"))

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Investigate window-sharing bias")
    parser.add_argument('--exps', nargs='*', type=int, default=[0])
    args = parser.parse_args()

    config_path = 'config.yaml'
    with open(config_path, 'r') as ymlfile:
        cfgs = yaml.safe_load(ymlfile)

    for exp in args.exps:
        run(exp, cfgs)