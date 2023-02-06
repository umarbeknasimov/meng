from environment import environment
from foundations import paths

def load_pretrained(output_location, pretrained_step, model, optimizer, scheduler):
    pretrained_model = environment.load(paths.model(output_location, pretrained_step))
    pretrained_optim = environment.load(paths.optim(output_location, pretrained_step))
    model.load_state_dict(pretrained_model)
    optimizer.load_state_dict(pretrained_optim['optimizer'])
    scheduler.load_state_dict(pretrained_optim['scheduler'])