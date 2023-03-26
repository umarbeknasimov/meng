import torch
import torch.nn.functional as F
from environment import environment
from foundations import paths
from foundations.hparams import ModelHparams
from models import cifar_resnet, cifar_resnet_layernorm, cifar_vgg

registered_models = [cifar_resnet.Model, cifar_vgg.Model, cifar_resnet_layernorm.Model]

def model_exists(save_location, save_step):
    return environment.exists(paths.model(save_location, save_step))

def optim_exists(save_location, save_step):
    return environment.exists(paths.optim(save_location, save_step))

def state_dicts_exist(save_location, save_step):
    return model_exists(save_location, save_step) and optim_exists(save_location, save_step)

def get_model_state_dict(output_location, step):
    return environment.load(paths.model(output_location, step))

def get_optim_state_dict(output_location, step):
    return environment.load(paths.optim(output_location, step))

def get(model_hparams: ModelHparams):
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_hparams.model_name):
            return registered_model.get_model_from_name(model_hparams.model_name)
    raise ValueError('No such model: {}'.format(model_hparams.model_name))

def get_default_hparams(model_name):
    """Get the default hyperparameters for a particular model."""

    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_name):
            params = registered_model.default_hparams()
            params.model_hparams.model_name = model_name
            return params

    raise ValueError('No such model: {}'.format(model_name))

def init_fn(w):
  if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
    torch.nn.init.kaiming_normal_(w.weight)
  if isinstance(w, torch.nn.BatchNorm2d):
    w.weight.data = torch.rand(w.weight.data.shape)
    w.bias.data = torch.zeros_like(w.bias.data)