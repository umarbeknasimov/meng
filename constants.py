import torch

USER_DIR = '/om/user/unasimov'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
