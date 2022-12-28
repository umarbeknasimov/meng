from utils import load, state_dict
import torch
import dataset
import torch.nn as nn
from utils import state_dict, interpolate
import models
import evaluate
import numpy as np
def get_projection(file1, file2, file3):
    resnet_1 = load.load_model_with_state(file1)
    resnet_2 = load.load_model_with_state(file2)
    resnet_center = load.load_model_with_state(file3)
    
    w_1 = state_dict.flatten_model_state_dict_w_o_num_batches(resnet_center)
    w_2 = state_dict.flatten_model_state_dict_w_o_num_batches(resnet_1)
    w_3 = state_dict.flatten_model_state_dict_w_o_num_batches(resnet_2)

    assert w_1.shape == w_2.shape
    assert w_2.shape == w_3.shape

    # vectors in plane
    u = (w_2 - w_1)
    v = (w_3 - w_1) - ((w_3 - w_1)@(w_2 - w_1))/(torch.norm(w_2 - w_1)**2) * (w_2 - w_1)

    u_hat = u / torch.norm(u)
    v_hat = v / torch.norm(v)

    return w_1, u_hat, v_hat

def get_3_model_comparison(w_1, u_hat, v_hat, range, device, losses_file=None, accs_file=None, eval_data='train'):
  x, y = torch.meshgrid([torch.arange(-10, 122, 5), torch.arange(-10, 122, 5)])
  x = x.to(device)
  y = y.to(device)
  train_data, valid_data = dataset.get_train_val_loaders()

  if eval_data == 'train':
    data_loader = train_data
  elif eval_data == 'valid':
    data_loader = valid_data
  else:
    raise ValueError()

  if losses_file == None:
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
      state_dict = state_dict.create_state_dict_w_o_batch_stats_from_x(P)
      model.load_state_dict(state_dict)
      interpolate.forward_pass(model)
      model.eval()

      loss, acc = evaluate.evaluate_data_loader(model, data_loader)
      
      all_losses[i][j] = loss
      all_accs[i][j] = acc
      print(f'-- x = {x_i}, y = {y_i} --')
      print(f'acc: {acc}, loss: {loss}')
      torch.save(all_losses, losses_file)
      torch.save(all_accs, accs_file)