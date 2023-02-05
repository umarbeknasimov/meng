import os

def logger(root): return os.path.join(root, 'logger')

def hparams(root): return os.path.join(root, 'hparams')

def state_dict(root, step): return os.path.join(root, f'state_dict__ep{step.ep}_it{step.it}.pth')

def model(root, step): return os.path.join(root, f'model__ep{step.ep}_it{step.it}.pth')

def train(root): return os.path.join(root, 'train')

def average(root): return os.path.join(root, 'average')

def spawn_instance(root, step, seed): return os.path.join(root, f'{step.ep}ep{step.it}it', f'seed__{seed}')

def spawn_average(root, step, seeds): 
    return os.path.join(root, f'{step.ep}ep{step.it}it', f'average__seeds{",".join([str(seed) for seed in seeds])}')