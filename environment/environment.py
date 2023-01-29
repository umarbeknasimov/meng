import os
import torch

def get_user_dir():
    return os.path.join('om', 'user', 'unasimov', 'new_framework')

def device():
    torch.device("cuda" if torch.cuda.is_available() else "cpu")