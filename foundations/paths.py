import os

def _step(step): return step.ep_it_str

def logger(root): return os.path.join(root, 'logger')

def plane_metrics(root): return os.path.join(root, 'plane_metrics')

def plane_grid(root): return os.path.join(root, 'plane_grid')

def checkpoint(root): return os.path.join(root, 'checkpoint.pth')

def hparams(root): return os.path.join(root, 'hparams')

def optim(root, step): return os.path.join(root, f'optim{_step(step)}.pth')

def model(root, step): return os.path.join(root, f'model{_step(step)}.pth')

def train(root): return os.path.join(root, 'train')

def spawn_step(root, step): return os.path.join(root, _step(step))

def seed(root, seed): return os.path.join(root, f'seed{seed}')

def average(root, seeds): 
    return os.path.join(root, f'average{",".join([str(seed) for seed in seeds])}')

def average_no_seeds(root): return os.path.join(root, 'average')