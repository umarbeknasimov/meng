from foundations import paths
from models.registry import get_model_state_dict, get_optim_state_dict

def load_pretrained(output_location, pretrained_step, model, optimizer, scheduler):
    pretrained_model = get_model_state_dict(output_location, pretrained_step)
    pretrained_optim = get_optim_state_dict(output_location, pretrained_step)
    model.load_state_dict(pretrained_model)
    optimizer.load_state_dict(pretrained_optim['optimizer'])
    scheduler.load_state_dict(pretrained_optim['scheduler'])