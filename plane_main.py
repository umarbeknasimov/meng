from plane import plane
from utils import load
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '/om/user/unasimov/models/'
weights1 = load.load(path + 'weights_frankle_seed_1_i=2048_seed_1', DEVICE)[-1]
weights2 = load.load(path + 'weights_frankle_seed_1_i=2048_seed_2', DEVICE)[-1]
weights3 = load.load(path + 'weights_frankle_seed_1_i=2048_seed_3', DEVICE)[-1]

losses_file = path + 'i=2048__losses'
accs_file = path + 'i=2048__accs'

w_1, u_hat, v_hat = plane.get_projection(weights1, weights2, weights3, DEVICE)

box_range = torch.arange(-10, 52, 2)

plane.get_3_model_comparison(w_1, u_hat, v_hat, box_range, DEVICE, losses_file, accs_file, True, 'valid')
