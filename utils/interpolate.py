import copy
import torch

from environment import environment

def interpolate_state_dicts_from_weights(weights1, weights2, alpha=0.5):
  model_1 = copy.deepcopy(weights1)
  model_2 = copy.deepcopy(weights2)
  new_state_dict = {}
  for param_tensor in model_1:
    new_state_dict[param_tensor] = alpha * model_2[param_tensor] + (1 - alpha) * model_1[param_tensor]
  return new_state_dict

def forward_pass(model, dataloader):
    # used for batch norm stats
    model.train()
    with torch.no_grad():
        for input, _ in dataloader:
            input_var = input.to(environment.device())
            # compute output
            model(input_var)