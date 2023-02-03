import os
import torch

def get_user_dir():
    path = os.path.join('/om', 'user', 'unasimov', 'runs')
    # if not os.path.exists(path):
    #     path = 'temp'
    return path

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open(file, mode='r'):
    return open(file, mode)

def exists(file):
    return os.path.exists(file)

def makedirs(path):
    return os.makedirs(path)

def save_model(model, path, *args, **kwargs):
    return torch.save(model, path, *args, **kwargs)

def load_model(path, *args, **kwargs):
    return torch.load(path, *args, **kwargs)