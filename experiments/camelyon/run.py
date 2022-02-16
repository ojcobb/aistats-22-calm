import torch
import numpy as np
import statsmodels.api as sm
import scipy
import pylab
import os
import argparse
import yaml
import wandb
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from copy import copy

from dd.utils import DictAsMember, merge_streams
from dd.loaders import load_detector, load_kernel, load_proj

from experiments.camelyon.single_run import compute_runtime
from experiments.camelyon.stream import stream_camelyon

def train(
    exp: int,
    cfgs: dict
):

    cfg = DictAsMember(cfgs[exp])
    stamped_exp = str(exp) + datetime.now().strftime("_%m-%d_%H:%M")
    save_dir = os.path.join('logs', stamped_exp)

    stream_p = stream_camelyon(split='train')
    qs = ['id_val', 'test']

    run = wandb.init(
        project='drift-detection-camelyon',
        name=stamped_exp,
        reinit=True,
        config=cfg,
        save_code=True
    ) 
    times_qs = {q: [] for q in qs}
    trial_means_qs = {q: [] for q in qs}
    cummean_qs = {q: 0 for q in qs}
    i_qs = {q: 0 for q in qs}

    for trial in range(cfg.n_trials):
        print(f"Trial {trial+1} of {cfg.n_trials}")

        initial_data = torch.stack([next(stream_p) for i in range(cfg.n_initial)], axis=0)
        
        if 'proj' in cfg:
            cfg_proj = deepcopy(cfg['proj'])
            cfg_proj_fit = cfg_proj.pop('fit')
            proj = load_proj(cfg_proj)
            if proj.configure:
                n_fit = int(initial_data.shape[0] * cfg_proj_fit['prop'])
                n_hold_out = initial_data.shape[0] - n_fit
                fitting_data, initial_data = torch.split(initial_data, [n_fit, n_hold_out])
                # Using ref data to configure proj "spends" it.
                proj.fit(fitting_data, cfg_proj_fit['epochs'], cfg_proj_fit['lr'], cfg_proj_fit['batch_size'])
            initial_data = torch.cat(
                [proj(x_i[None,:]) for x_i in tqdm(initial_data, "Projecting ref data")], axis=0
            )
        else:
            proj = lambda x: x


        args = [initial_data, cfg.ert]
        kwargs = copy(cfg['detector'])
        if 'kernel' in cfg:
            kernel = load_kernel(cfg.kernel)
            args = [kernel] + args
        detector = load_detector(args, kwargs)

        for q in ['id_val', 'test']:

            stream_q = stream_camelyon(split=q)
            subtimes = []
            for _ in tqdm(range(cfg.times_per_trial)):
                stream = merge_streams(stream_p, stream_q, cfg.change_point)
                subtime = compute_runtime(
                    detector, stream, cfg.change_point, proj=proj
                )
                subtimes.append(subtime)
                cummean_qs[q] = i_qs[q]*cummean_qs[q]/(i_qs[q]+1) + subtime/(i_qs[q]+1)
                i_qs[q] +=1
                
                logs = {f"cummean_{q}": cummean_qs[q]}
                if i_qs[q] % cfg.log_mean_every == 0:
                    wandb.log(logs)

            times_qs[q] += subtimes
            trial_means_qs[q] += np.array(subtimes).mean()
            print(f'Trial {trial+1} average time for {q}: {np.array(subtimes).mean()}')
            wandb.log({f'trial_time_{q}': np.array(subtimes).mean()})

        if (trial+1) % cfg.save_every == 0:
            save_subdir = os.path.join(save_dir, str(trial))
            if not os.path.isdir(save_subdir):
                os.makedirs(save_subdir)
            for q in qs:  
                plot_path = os.path.join(save_subdir, f'qqplot_{q}.png')
                sm.qqplot(np.array([t+1 for t in times_qs[q]]), scipy.stats.geom(1/cfg.ert), line='45')
                pylab.savefig(plot_path)
                np.save(os.path.join(save_subdir, f'times_{q}.npy'), np.array(times_qs[q]))
                with open(os.path.join(save_subdir, 'config.yaml'), 'w+') as fd:
                    yaml.dump(dict(cfg), fd)

        del detector
        
    for q in qs:
        plot_path = os.path.join(save_dir, f'qqplot_{q}.png')  
        sm.qqplot(np.array([t+1 for t in times_qs[q]]), scipy.stats.geom(1/cfg.ert), line='45')
        pylab.savefig(plot_path)
        np.save(os.path.join(save_dir, f'times_{q}.npy'), np.array(times_qs[q]))
        with open(os.path.join(save_dir, 'config.yaml'), 'w+') as fd:
            yaml.dump(dict(cfg), fd)
        logs.update({f'final_avg_{q}': np.array(times_qs[q]).mean()})
        logs.update({f'trial_means_{q}': np.array(trial_means_qs[q])})
        logs.update({f'qqplot_{q}': wandb.Image(plot_path)})
        print(f"Final average time for {q}: {np.array(times_qs[q]).mean()}")        
    wandb.log(logs)

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run online drift detection")
    parser.add_argument('--exps', nargs='*', type=int, default=[0])
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    config_path = 'config.yaml'
    with open(config_path, 'r') as ymlfile:
        cfgs = yaml.safe_load(ymlfile)

    for exp in args.exps:
        train(exp, cfgs)