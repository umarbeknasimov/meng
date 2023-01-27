import torch

USER_DIR = '/om/user/unasimov'
EXPONENTIAL_STEPS = [2**i for i in range(20)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
