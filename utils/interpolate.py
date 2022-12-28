import copy
import torch

def interpolate_state_dicts_from_weights(weights1, weights2, alpha=0.5):
  model_1 = copy.deepcopy(weights1)
  model_2 = copy.deepcopy(weights2)
  new_state_dict = {}
  for param_tensor in model_1:
    new_state_dict[param_tensor] = alpha * model_2[param_tensor] + (1 - alpha) * model_1[param_tensor]
  return new_state_dict

def forward_pass(model, dataloader, device):
    # used for batch norm stats
    model.train()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            input_var = input.to(device)
            # compute output
            model(input_var)