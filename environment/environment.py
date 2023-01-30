import os
import torch

def get_user_dir():
    path = os.path.join('/om', 'user', 'unasimov', 'new_framework')
    if not os.path.exists(path):
        path = 'temp'
    return path

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")