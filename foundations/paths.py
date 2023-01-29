import os

def logger(root): return os.path.join(root, 'logger')

def hparams(root): return os.path.join(root, 'hparams')

def state_dict(root, step): return os.path.join(root, f'ep{step.ep}_it{step.it}')

def train(root, seed, init_step=None, init_step_seed=None):
    train_details = f's_{seed}'
    if init_step and init_step_seed:
        train_details = os.path.join(f'init_ep{init_step.ep}_it{init_step.it}_s{init_step_seed}', train_details)
    return os.path.join(root, 'train', train_details)