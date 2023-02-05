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

def average_state_dicts(weights):
  if len(weights) <= 0:
    raise ValueError(f'can only average > 0 weights but got {len(weights)} weights')
  new_state_dict = {}
  for param_tensor in weights[0]:
    new_state_dict[param_tensor] = torch.mean(torch.stack([weight[param_tensor] for weight in weights]), dim=0)
  return new_state_dict

def forward_pass(model, dataloader):
    # used for batch norm stats
    model.train()
    with torch.no_grad():
        for input, _ in dataloader:
            input_var = input.to(environment.device())
            # compute output
            model(input_var)