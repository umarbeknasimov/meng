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
  for param_name in weights[0]:
    stacked_weights = torch.stack([weight[param_name] for weight in weights], dim=0).to(torch.float64)
    new_state_dict[param_name] = torch.mean(stacked_weights, dim=0)
  return new_state_dict

def average_optimizer_state_dicts(state_dicts):
  if len(state_dicts) <= 0:
    raise ValueError(f'can only average > 0 state_dicts but got {len(state_dicts)} state_dicts')
  new_state_dict = {}
  new_state_dict['param_groups'] = copy.deepcopy(state_dicts[0]['param_groups'])
  new_state_dict['state'] = {}
  for i, _ in state_dicts[0]['state'].items():
    new_state_dict_i = {}
    for param_name in state_dicts[0]['state'][0]: # checking for param_name, in our case is just 'momentum_buffer'
      if state_dicts[0]['state'][0][param_name] is not None:
        stacked_weights = torch.stack([state_dict['state'][i][param_name] for state_dict in state_dicts], dim=0).to(torch.float64)
        new_state_dict_i[param_name] = torch.mean(stacked_weights, dim=0)
      else:
        new_state_dict_i[param_name] = None
    new_state_dict['state'][i] = new_state_dict_i
  return new_state_dict

def forward_pass(model, dataloader):
    # used for batch norm stats
    model.train()
    with torch.no_grad():
        for input, _ in dataloader:
            input_var = input.to(environment.device())
            # compute output
            model(input_var)