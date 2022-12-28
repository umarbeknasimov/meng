import numpy as np
from utils import load
import evaluate
import json
import torch.nn as nn
import dataset

def interpolate_weights_at_all_epochs(file1, name, device, file2=None, eval_data='valid'):
  # apply interpolation between weights
  # if file2 is none, then this is interpolation for single model: iteration i -> final
  # if file2 is not none, then this is iterpolation for 2 models at every epoch
  # alpha_range = np.arange(0, 1.05, 0.05)
  alpha_range = np.arange(0, 1.05, 0.1)
  train_data, valid_data = dataset.get_train_val_loaders()

  model_weights_1 = load(file1)
  if file2 != None:
    model_weights_2 = load(file2)
    print(f'model 1 has {len(model_weights_1)} weights and model 2 has {len(model_weights_2)} weights')
    assert len(model_weights_1) == len(model_weights_2), 'model weights sizes are different'
  iteration_range = np.arange(0, len(model_weights_1))

  if eval_data == 'train':
    data_loader = train_data
  elif eval_data == 'valid':
    data_loader = valid_data
  else:
    raise ValueError()
  
  try:
    with open(name, 'r') as f:
      print('getting interp info')
      stats = json.loads(json.load(f))
      stats['losses'] = [[float(val_i) for val_i in val] for val in stats['losses']]
      stats['accs'] = [[float(val_i) for val_i in val] for val in stats['accs']]
      print(f'there are {len(stats["losses"])} already filled values')
  except:
    print('could not find interp file, starting fresh')
    stats = {
        'losses': [],
        'accs': []
    }

  criterion = nn.CrossEntropyLoss()

  for i in iteration_range:
    if len(stats['losses']) > i:
      print(f'skipping {i}')
      continue
    print(f"getting max loss decrease for iteration {i}")
    all_losses = []
    all_accs = []
    for alpha in alpha_range:
      if file2 != None:
        loss, acc = evaluate.eval_interpolation(model_weights_1[i], model_weights_2[i], alpha, data_loader, device, True)
      else:
        loss, acc = evaluate.eval_interpolation(model_weights_1[i], model_weights_1[-1], alpha, data_loader, device, True)
      all_losses.append(loss)
      all_accs.append(acc)
    stats['losses'].append(all_losses)
    stats['accs'].append(all_accs)
    if file2 != None:
      max_loss_dec = min([all_losses[0], all_losses[-1]]) - min(all_losses)
      max_acc_inc = max(all_accs) - max([all_accs[0], all_accs[-1]])
      print(f"for iteration {i}, max loss decrease is {max_loss_dec}, max acc increase is {max_acc_inc}")
    else:
      max_loss_inc = max(all_losses) - all_losses[0]
      max_acc_dec = min(all_accs) - all_accs[0]
      print(f"for iteration {i}, max loss increase is {max_loss_inc}, max acc decrease is {max_acc_dec}")

    with open(name, 'w') as f:
        json_string = json.dumps(stats)
        json.dump(json_string, f)