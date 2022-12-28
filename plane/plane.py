from utils import load, state_dict
import torch
import dataset
from utils import state_dict, interpolate
import models
import evaluate
import numpy as np
def get_projection(weights1, weights2, weights3, device):
    resnet_1 = load.model_with_state(weights1, device)
    resnet_2 = load.model_with_state(weights2, device)
    resnet_center = load.model_with_state(weights3, device)
    
    w_1 = state_dict.flatten_model_params(resnet_center)
    w_2 = state_dict.flatten_model_params(resnet_1)
    w_3 = state_dict.flatten_model_params(resnet_2)

    assert w_1.shape == w_2.shape
    assert w_2.shape == w_3.shape

    # vectors in plane
    u = (w_2 - w_1)
    v = (w_3 - w_1) - ((w_3 - w_1)@(w_2 - w_1))/(torch.norm(w_2 - w_1)**2) * (w_2 - w_1)

    u_hat = u / torch.norm(u)
    v_hat = v / torch.norm(v)

    return w_1, u_hat, v_hat

def get_3_model_comparison(w_1, u_hat, v_hat, box_range, device, losses_file, accs_file, start_new=True, eval_data='train'):
  x, y = torch.meshgrid([box_range, box_range])
  x = x.to(device)
  y = y.to(device)
  train_data, valid_data = dataset.get_train_val_loaders()

  if eval_data == 'train':
    data_loader = train_data
  elif eval_data == 'valid':
    data_loader = valid_data
  else:
    raise ValueError()

  if start_new:
    print('starting fresh')
    all_losses = np.zeros(x.shape)
    all_accs = np.zeros(x.shape)
  else:
    all_losses = torch.load(losses_file)
    all_accs = torch.load(accs_file)

  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      if all_losses[i][j] != 0 and all_accs[i][j] != 0:
        print(f'skipping {i}, {j} because already computed')
        continue
      x_i = x[i][j]
      y_i = y[i][j]
      P = w_1 + x_i * u_hat + y_i * v_hat
      model = models.frankleResnet20().to(device)
      state_dict_new = state_dict.create_state_dict_w_o_batch_stats_from_x(P)
      model.load_state_dict(state_dict_new)
      interpolate.forward_pass(model, data_loader, device)
      model.eval()

      loss, acc = evaluate.evaluate_data_loader(model, data_loader, device)
      
      all_losses[i][j] = loss
      all_accs[i][j] = acc
      print(f'-- x = {x_i}, y = {y_i} --')
      print(f'acc: {acc}, loss: {loss}')
      torch.save(all_losses, losses_file)
      torch.save(all_accs, accs_file)