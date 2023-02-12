from environment import environment
from foundations import paths
from foundations.hparams import ModelHparams
from models.cifar_resnet import Model

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
    return Model.get_model_from_name(model_hparams.model_name)

def load_pretrained_model(pretrained_output_location, pretrained_step, model):
    model_state_dict = get_model_state_dict(pretrained_output_location, pretrained_step)
    model.load_state_dict(model_state_dict)