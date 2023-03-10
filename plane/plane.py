from utils import state_dict
import torch
from utils import state_dict, interpolate, evaluate
import models.registry
import numpy as np
import datasets.registry
from environment import environment
from foundations import paths

def get_plane_bases(weights1, weights2, weights3):
  w_1 = state_dict.flatten_state_dict_w_o_batch_stats(weights1)
  w_2 = state_dict.flatten_state_dict_w_o_batch_stats(weights2)
  w_3 = state_dict.flatten_state_dict_w_o_batch_stats(weights3)

  assert w_1.shape == w_2.shape
  assert w_2.shape == w_3.shape

  # vectors in plane
  u = (w_2 - w_1)
  v = (w_3 - w_1) - ((w_3 - w_1)@(w_2 - w_1))/(torch.norm(w_2 - w_1)**2) * (w_2 - w_1)

  u_hat = u / torch.norm(u)
  v_hat = v / torch.norm(v)

  return w_1, w_2, w_3, u_hat, v_hat

def get_x_y(point, origin, u_hat, v_hat):
  return torch.dot(point - origin, u_hat).item(), torch.dot(point - origin, v_hat)

def evaluate_plane(weights1, weights2, weights3, output_location, model_hparams, dataset_hparams, steps=10):
  w_1, w_2, w_3, u_hat, v_hat = get_plane_bases(weights1, weights2, weights3)

  w_2_proj_x, w_2_proj_y = get_x_y(w_2, w_1, u_hat, v_hat)
  w_3_proj_x, w_3_proj_y = get_x_y(w_3, w_1, u_hat, v_hat)
  max_x, max_y = max(w_2_proj_x, w_3_proj_x), max(w_2_proj_y, w_3_proj_y)
  print(f'max_x {max_x}, max_y {max_y}')

  edge_factor = 0.2
  x_start, x_end = 0 - max_x * edge_factor, max_x * (1 + edge_factor)
  y_start, y_end = 0 - max_y * edge_factor, max_y * (1 + edge_factor)

  x_range = torch.linspace(x_start, x_end, steps=steps).to(environment.device())
  y_range = torch.linspace(y_start, y_end, steps=steps).to(environment.device())

  print('using x range: ', x_range)
  print('using y range: ', y_range)

  x, y = torch.meshgrid((x_range, y_range), indexing='ij')

  torch.save({'x': x, 'y': y}, paths.plane_grid(output_location))
  train_data, test_data = datasets.registry.get(dataset_hparams), datasets.registry.get(dataset_hparams, False)

  metrics_shape = (steps, steps)

  if environment.exists(paths.plane_metrics(output_location)):
    metrics = torch.load(paths.plane_metrics(output_location))
  else:
    metrics = {
      'train_loss': np.zeros(metrics_shape),
      'train_accuracy': np.zeros(metrics_shape),
      'test_loss': np.zeros(metrics_shape),
      'test_accuracy': np.zeros(metrics_shape)
    }

  for data_set, dataloader in {'train': train_data, 'test': test_data}.items():
    loss_name = f'{data_set}_loss'
    accuracy_name = f'{data_set}_accuracy'
    for i in range(steps):
      for j in range(steps):
        if metrics[loss_name][i][j] != 0 and metrics[accuracy_name][i][j] != 0:
          print(f'skipping {i}, {j} because already computed')
          continue
        x_i = x[i][j]
        y_i = y[i][j]
        P = w_1 + x_i * u_hat + y_i * v_hat
        model = models.registry.get(model_hparams)
        state_dict_new = state_dict.create_state_dict_w_o_batch_stats_from_x(model, P)
        model.load_state_dict(state_dict_new)
        interpolate.forward_pass(model, train_data)
        model.eval()
        loss, acc = evaluate.evaluate(model, dataloader)
        
        metrics[loss_name][i][j] = loss
        metrics[accuracy_name][i][j] = acc
        print(f'x = {x_i:8.2f}, y = {y_i:8.2f}: acc: {acc:4.2f}, loss: {loss:4.4f}')

        torch.save(metrics, paths.plane_metrics(output_location))