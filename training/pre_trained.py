import torch

from foundations import paths

def load_pretrained(output_location, pretrained_step, model, optimizer, scheduler):
    pretrained_state_dict = torch.load(paths.state_dict(output_location, pretrained_step))
    model.load_state_dict(pretrained_state_dict['model'])
    optimizer.load_state_dict(pretrained_state_dict['optimizer'])
    scheduler.load_state_dict(pretrained_state_dict['scheduler'])

def get_pretrained_model(output_location, pretrained_step):
    pretrained_state_dict = torch.load(paths.state_dict(output_location, pretrained_step))
    return pretrained_state_dict['model']