import os

def _step(step): return step.ep_it_str

def logger(root): return os.path.join(root, 'logger')

def ids_logger(root): return os.path.join(root, 'ids_logger')

def plane_metrics(root): return os.path.join(root, 'plane_metrics')

def lookahead_metrics(root, seed): return os.path.join(root, f'lookahead_metrics_{seed}')

def distances_metrics(root): return os.path.join(root, f'distances_metrics')

def plane_grid(root): return os.path.join(root, 'plane_grid')

def checkpoint(root): return os.path.join(root, 'checkpoint.pth')

def ema(root): return os.path.join(root, f'ema.pth')

def ema_warm(root, step): return os.path.join(root, f'ema_warm{_step(step)}.pth')

def hparams(root): return os.path.join(root, 'hparams')

def optim(root, step): return os.path.join(root, f'optim{_step(step)}.pth')

def model(root, step): return os.path.join(root, f'model{_step(step)}.pth')

def train(root): return os.path.join(root, 'train')

def step_(root, step): return os.path.join(root, _step(step))

def seed(root, seed): return os.path.join(root, f'seed{seed}')

def seeds(root, seeds): 
    return os.path.join(root, f'seeds{",".join([str(seed) for seed in seeds])}')

def legs(root): return os.path.join(root, 'legs')

def average(root, seeds): 
    return os.path.join(root, f'average{",".join([str(seed) for seed in seeds])}')

def average_no_seeds(root): return os.path.join(root, 'average')