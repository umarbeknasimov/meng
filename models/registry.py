import torch

from environment import environment
from foundations import paths

def exists(save_location, save_step):
    return environment.exists(paths.state_dict(save_location, save_step)) or environment.exists(paths.model(save_location, save_step))

def get_model_state_dict(output_location, step):
    model_state_dict = torch.load(paths.state_dict(output_location, step))
    return model_state_dict['model']