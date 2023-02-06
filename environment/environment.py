import os
import torch

def get_user_dir():
    path = os.path.join('/om', 'user', 'unasimov', 'runs')
    if not os.path.exists(path):
        path = 'temp'
    return path

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open(file, mode='r'):
    return open(file, mode)

def exists(file):
    return os.path.exists(file)

def makedirs(path):
    return os.makedirs(path)

def exists_or_makedirs(path):
    if not exists(path):
        return makedirs(path)
    return True

def save(obj, path, *args, **kwargs):
    return torch.save(obj, path, *args, **kwargs)

def load(path, *args, **kwargs):
    return torch.load(path, *args, **kwargs)