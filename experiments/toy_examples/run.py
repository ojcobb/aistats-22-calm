import torch
import numpy as np
import statsmodels.api as sm
import scipy
import pylab
import os
from itertools import product
import yaml
import wandb
from datetime import datetime
import tqdm
from copy import deepcopy, copy

from dd.utils import DictAsMember, merge_streams

from experiments.toy_examples.single_run import compute_runtime
from experiments.toy_examples.stream import load_stream
from experiments.toy_examples.loaders import load_detector, load_kernel, load_proj


def train(
    cfg: dict
):

    cfg = DictAsMember(cfg)
    stamped_exp = str(exp) + datetime.now().strftime("_%m-%d_%H%M")
    save_dir = os.path.join('logs', stamped_exp)

    stream_p = load_stream(cfg['P'])
    stream_qs = [load_stream(cfg_q) for cfg_q in cfg['Q'].values()]
    n_qs = len(stream_qs)

    run = wandb.init(
        project='drift-detection-paper-new',
        name=stamped_exp,
        reinit=True,
        config=cfg,
        save_code=True
    )  
    times_qs = {key: [] for key in cfg['Q'].keys()}
    trial_means_qs = {key: [] for key in cfg['Q'].keys()}
    cummean_qs = {key: 0 for key in cfg['Q'].keys()}
    i_qs = {key: 0 for key in cfg['Q'].keys()}

    for trial in range(cfg.n_trials):
        print(f"Trial {trial+1} of {cfg.n_trials}")

        initial_data = torch.stack([next(stream_p) for i in range(cfg.n_initial)], axis=0)
        
        if 'proj' in cfg:
            cfg_proj = deepcopy(cfg['proj'])
            config_prop = cfg_proj.pop('config_prop', 0)
            proj = load_proj(cfg_proj)
            if proj.configure:
                n_fit = int(initial_data.shape[0] * config_prop)
                n_hold_out = initial_data.shape[0] - n_fit
                fitting_data, initial_data = torch.split(initial_data, [n_fit, n_hold_out])
                # Using ref data to configure proj "spends" it.
                proj.fit(fitting_data)
            initial_data = torch.stack([proj(x_i[None,:]) for x_i in initial_data], axis=0)
        else:
            proj = lambda x: x


        args = [initial_data, cfg.ert]
        kwargs = copy(cfg['detector'])
        if 'kernel' in kwargs:
            kernel = load_kernel(kwargs.pop('kernel'))
            args = [kernel] + args
        detector = load_detector(args, kwargs) # Detector is configured here

        for q, cfg_q in cfg['Q'].items():
            stream_q = load_stream(cfg_q)
            subtimes = []
            for _ in tqdm.tqdm(range(cfg.times_per_trial)):
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

    for q in cfg['Q'].keys():       
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
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

    np.random.seed(42)
    torch.manual_seed(42)

    gen_config_path = 'gen_config.yaml'
    with open(gen_config_path, 'r') as ymlfile:
        gen_cfgs = yaml.safe_load(ymlfile)

    data_config_path = 'data_config.yaml'
    with open(data_config_path, 'r') as ymlfile:
        data_cfgs = yaml.safe_load(ymlfile)

    detector_config_path = 'detector_config.yaml'
    with open(detector_config_path, 'r') as ymlfile:
        detector_cfgs = yaml.safe_load(ymlfile)

    all_exps = product(data_cfgs.keys(), gen_cfgs.keys(), detector_cfgs.keys())
    
    for exp in all_exps:
        cfg = gen_cfgs[exp[1]]
        cfg.update(detector=detector_cfgs[exp[2]])
        cfg.update(P=data_cfgs[exp[0]]['P'])
        cfg.update(Q=data_cfgs[exp[0]]['Q'])
        train(deepcopy(cfg))