"""
Some useful tools.
"""
import os
import math
import json
import random
import numpy as np
import torch
from contextlib import contextmanager


@contextmanager
def ignored(name):
    """
    use this context manager for try/except functions
    that just pass exceptions.

    args:
    :param name: name of the exception

    Example:
        with ignored(OSError):
            os.makedirs(dir_name)

    """
    try:
        yield
    except name:
        pass


def get_overload_parameters(args, argument_list):
    """
    iterates over argument list and appends yacs argument name
    and corresponding value if it's not None

    args:
    :param args: argparser Namespace
    :param argument_list: list of tuples in the format
                            (parser argument name, yacs argument name)

    returns: list with yacs argument name and corresopnding value
    """
    args = vars(args)
    parameters_list = []

    for argument, param in argument_list:
        if args[argument] is not None:
            parameters_list.append(param)
            parameters_list.append(args[argument])

    return parameters_list


def set_random_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_dir(work_dir, new_dir):
    file_name = os.path.join(work_dir, new_dir)

    with ignored(OSError):
        os.makedirs(file_name)

    return file_name


def get_checkpoint_folder(dataset_chkp, chkp_prefix, cfg):
    new_dir_name = '{chkp_prefix}_depth{depth}_latent{latent}_seed{seed}'.format(
        chkp_prefix=chkp_prefix,
        depth=cfg.MODEL.DEPTH,
        latent=cfg.MODEL.LATENT,
        seed=cfg.SOLVER.SEED)

    return create_dir(dataset_chkp, new_dir_name)


def _check_eval_in_files(files_with_prefix, selected_epoch):
    selected_epoch_files = []
    selected_epoch = selected_epoch + 1

    while not any('eval' in file for file in selected_epoch_files):
        selected_epoch = selected_epoch - 1

        # get all file names containing the epoch
        selected_epoch_files = [fname for fname in files_with_prefix
                                if str(selected_epoch) in fname]
        if selected_epoch == 0:
            break
    return selected_epoch_files, selected_epoch


def load_models(models_dict, cfg, epoch='max'):

    files_with_prefix = [fname for fname in os.listdir(cfg.DIRS.CHKP_DIR)
                         if cfg.DIRS.CHKP_PREFIX in fname]

    try:
        assert (len(files_with_prefix) != 0)
    except AssertionError:
        print(f'--- NO FILES WITH PREFIX "{cfg.DIRS.CHKP_PREFIX}" FOUND. ---\n'+
              '--- SKIPPING MODEL LOADING. STARTING FROM EPOCH 1. ---')

        cfg.SOLVER.COMPLETED_EPOCHS = 0
        return models_dict, cfg

    available_epochs = [int(f.split('_')[-1].split('.')[0]) for
                        f in files_with_prefix]

    selected_epoch = sorted(available_epochs)[-1] \
        if epoch == 'max' else int(epoch)

    # check if ignite eval file exists for epoch
    # (often doesn't exist if script exits during val phase)
    # if not, select a previous checkpoint where it does exist.
    # if no such checkpoint exists, don't load models
    try:
        selected_epoch_files, selected_epoch = \
            _check_eval_in_files(files_with_prefix, selected_epoch)
        cfg.SOLVER.COMPLETED_EPOCHS = int(selected_epoch)
        assert (selected_epoch != 0)
    except AssertionError:
        print('--- NO IGNITE EVAL FILES FOUND. ---')
        print('--- STARTING FROM EPOCH 1. ---')
        return models_dict, cfg

    for model_name, model in models_dict.items():
        file_name = [fname for fname in selected_epoch_files
                     if model_name in fname]

        error_msg = f'Either several (> 1) or no files found for ' + \
                    f'model: {model_name} , epoch: {selected_epoch}.'
        assert (len(file_name) == 1), error_msg

        file_path = os.path.join(cfg.DIRS.CHKP_DIR, file_name[0])
        model.load_state_dict(torch.load(file_path))

        if 'opt' in model_name:
            for state in model.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.MODEL.DEVICE)

        models_dict[model_name] = model

    cfg = load_ignite_params(cfg.DIRS.CHKP_DIR,
                              selected_epoch_files, cfg)

    return models_dict, cfg


def save_ignite_params(engine, engine_name, cfg):
    epoch = engine.state.epoch if engine_name == 'trainer' \
        else cfg.SOLVER.EVAL_EPOCH

    ignite_params = {
        f'{engine_name}_iteration': engine.state.iteration,
        f'{engine_name}_epoch': epoch}

    file_name = '{}_ignite_{}_{}.pth'.format(
        cfg.DIRS.CHKP_PREFIX,
        engine_name,
        epoch)

    abs_name = os.path.join(cfg.DIRS.CHKP_DIR, file_name)

    torch.save(ignite_params, abs_name)

    return None


def load_ignite_params(checkpoint_folder, files, cfg):
    for file in files:
        if 'trainer' in file:
            file_name = os.path.join(checkpoint_folder, file)

            params = torch.load(file_name)
            cfg.SOLVER.TRAINER_ITERATION = params[f'trainer_iteration']
            cfg.SOLVER.TRAINER_EPOCH = params[f'trainer_epoch']

        elif 'eval' in file:
            file_name = os.path.join(checkpoint_folder, file)

            params = torch.load(file_name)
            cfg.SOLVER.EVAL_ITERATION = params[f'eval_iteration']
            cfg.SOLVER.EVAL_EPOCH = params[f'eval_epoch']

    return cfg


def swapaxes(x, a, b):
    try:
        return x.swapaxes(a, b)
    except AttributeError:  # support pytorch
        return x.transpose(a, b)


# pytorch doesn't support negative strides / can't flip tensors
# so instead this function swaps the two halves of a tensor
def swap_halves(x):
    split_list = list(x.split(x.shape[0] // 2))
    split_list.reverse()
    return torch.cat(split_list)


# torch.lerp only support scalar weight
def lerp(start, end, weights):
    return start + weights * (end - start)


def L2(x):
    return torch.mean(x ** 2)


def find_rectangle(n):
    max_side = int(math.sqrt(n))
    for h in range(2, max_side + 1)[::-1]:
        w = n // h
        if (h * w) == n:
            return (h, w)
    return n, 1


def make_mosaic(x, nx=None, ny=None):

    n, h, w = x.shape[:3]
    has_channels = len(x.shape) > 3

    if has_channels:
        c = x.shape[3]

    if nx is None and ny is None:
        ny, nx = find_rectangle(n)
    elif ny is None:
        ny = n // nx
    elif nx is None:
        nx = n // ny

    end_shape = (w, c) if has_channels else (w,)
    mosaic = x.reshape(ny, nx, h, *end_shape)
    mosaic = swapaxes(mosaic, 1, 2)

    hh = mosaic.shape[0] * mosaic.shape[1]
    ww = mosaic.shape[2] * mosaic.shape[3]
    end_shape = (ww, c) if has_channels else (ww,)
    mosaic = mosaic.reshape(hh, *end_shape)
    return mosaic


def calc_mean_non_empty(cfg_name, cfg_list):
    """

    :param cfg_name:
    :param cfg_list:
    :return:
    """
    if len(cfg_list) != 0:
        avg = str(sum(cfg_list) / float(len(cfg_list)))
        return '{name}={avg} | '.format(name=cfg_name, avg=avg)
    else:
        return ''
