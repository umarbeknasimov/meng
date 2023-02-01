import os

def logger(root): return os.path.join(root, 'logger')

def hparams(root): return os.path.join(root, 'hparams')

def state_dict(root, step): return os.path.join(root, f'ep{step.ep}_it{step.it}.pth')

def train(root): return os.path.join(root, 'train')