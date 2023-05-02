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

  non_none_state_dict = None
  for state_dict in state_dicts:
    if len(state_dict['state'].keys()) > 0:
      non_none_state_dict = state_dict
      break
  
  if non_none_state_dict == None: return new_state_dict
  
  for weights_index in non_none_state_dict['state'].keys():
    optim_param_for_weights_index = {}
    for optim_param_name, optim_param in non_none_state_dict['state'][weights_index].items():
      optim_param_for_weights_index[optim_param_name] = torch.zeros(optim_param.size())
    new_state_dict[weights_index] = optim_param_for_weights_index

  for weights_index in non_none_state_dict['state'].keys():
    optim_param_for_weights_index = {}
    for optim_param_name, _ in non_none_state_dict['state'][weights_index].items(): # checking for param_name, in our case is just 'momentum_buffer'
      stacked_weights = []
      for state_dict in state_dicts:
        if weights_index in state_dict['state']:
          stacked_weights.append(state_dict['state'][weights_index][optim_param_name])
      
      optim_param_for_weights_index[optim_param_name] = torch.mean(torch.stack(stacked_weights, dim=0).to(torch.float64), dim=0)
    new_state_dict['state'][weights_index] = optim_param_for_weights_index
  return new_state_dict

def forward_pass(model, dataloader):
    # used for batch norm stats
    model.train()
    with torch.no_grad():
        for input, _ in dataloader:
            input_var = input.to(environment.device())
            # compute output
            model(input_var)