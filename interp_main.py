from interpolate import interpolate
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '/om/user/unasimov/models/'
file1 = path + 'weights_frankle_seed_1_i=2048_seed_1'
file2 = path + 'weights_frankle_seed_1_i=2048_seed_2'

losses_file = path + 'i=2048__losses'
accs_file = path + 'i=2048__accs'

name = 'i=2048_interp__train_data'

interpolate.interpolate_weights_at_all_epochs(file1, name, DEVICE, file2, 'train')