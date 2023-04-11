import numpy as np
import torch
from environment import environment
from foundations import paths
import models.registry
import datasets.registry
from spawning.runner import SpawningRunner
from utils import interpolate, state_dict, evaluate


def compute_lookahead(spawning_runner: SpawningRunner, child_data_order_seed: int):
  assert child_data_order_seed in spawning_runner.children_data_order_seeds
  parent_steps = spawning_runner.desc.saved_steps
  children_steps = spawning_runner.desc.saved_steps
  parent_location = spawning_runner.train_location()

  output_location = spawning_runner.desc.run_path(part='lookahead', experiment=spawning_runner.experiment)
  environment.exists_or_makedirs(output_location)

  metrics_shape = (len(parent_steps), len(children_steps))

  if environment.exists(paths.lookahead_metrics(output_location, child_data_order_seed)):
    metrics = torch.load(paths.lookahead_metrics(output_location, child_data_order_seed))
  else:
    metrics = {
      'train_loss': np.zeros(metrics_shape),
      'train_accuracy': np.zeros(metrics_shape),
      'test_loss': np.zeros(metrics_shape),
      'test_accuracy': np.zeros(metrics_shape)
    }

  train_data, test_data = datasets.registry.get(spawning_runner.desc.dataset_hparams), datasets.registry.get(spawning_runner.desc.dataset_hparams, False)


  for i, parent_step in enumerate(parent_steps):
    parent_state_dict = environment.load(paths.model(parent_location, parent_step))
    child_location = spawning_runner.spawn_step_child_location(parent_step, child_data_order_seed)
    for j, child_step in enumerate(children_steps):
        child_state_dict = environment.load(paths.model(child_location, child_step))
        weights = [parent_state_dict, child_state_dict]
        model = models.registry.get(spawning_runner.desc.model_hparams).to(environment.device())
        averaged_weights = interpolate.average_state_dicts(weights)
        averaged_weights_wo_batch_stats = state_dict.get_state_dict_wo_batch_stats(
            model, averaged_weights)
        for data_set, dataloader in {'train': train_data, 'test': test_data}.items():
            loss_name = f'{data_set}_loss'
            accuracy_name = f'{data_set}_accuracy'
            if metrics[loss_name][i][j] != 0 and metrics[accuracy_name][i][j] != 0:
               continue
            model = models.registry.get(spawning_runner.desc.model_hparams).to(environment.device())
            model.load_state_dict(averaged_weights_wo_batch_stats)
            interpolate.forward_pass(model, train_data)
            model.eval()
            loss, acc = evaluate.evaluate(model, dataloader)
            metrics[loss_name][i][j] = loss
            metrics[accuracy_name][i][j] = acc
            print(f'i = {i:8.2f}, j = {j:8.2f}: acc: {acc:4.2f}, loss: {loss:4.4f}')
        torch.save(metrics, paths.lookahead_metrics(output_location, child_data_order_seed))