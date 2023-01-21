
from utils import average, interpolate, state_dict, interpolate
import models
import torch.nn as nn
import torch
import copy

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # print(topk)
    maxk = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate_data_loader(model, data_loader, device):
  losses = average.AverageMeter()
  top1 = average.AverageMeter()
  criterion = nn.CrossEntropyLoss()
  correct_ids = []
  ids_multiplier_offset = data_loader.batch_size

  with torch.no_grad():
    for i, (input, target) in enumerate(data_loader):
      target = target.to(device)
      input_var = input.to(device)
      target_var = target

      # compute output
      output = model(input_var)
      loss = criterion(output, target_var)

      output = output.float()
      loss = loss.float()
      # measure accuracy and record loss
      prec1 = accuracy(output.data, target)[0]
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      
      curr_ids_correct = ((output.data.argmax(-1) == target) == True).nonzero().squeeze()
      correct_ids.extend((curr_ids_correct + ((torch.ones(curr_ids_correct.shape) * i * ids_multiplier_offset))).tolist())
  return losses.avg, top1.avg, correct_ids

def eval_interpolation(weights1, weights2, alpha, data_loader, device, warm):
  # model_1 centered at x = 0, model_2 centered at x = 1
  new_state_dict = interpolate.interpolate_state_dicts_from_weights(weights1, weights2, alpha)

  model = models.frankleResnet20().to(device)
  if warm:
    new_state_dict = state_dict.get_state_dict_w_o_batch_stats(new_state_dict)
    model.load_state_dict(new_state_dict)
    print('running forward pass')
    interpolate.forward_pass(model, data_loader, device)
  else:
    new_state_dict = state_dict.get_state_dict_w_o_num_batches_tracked(new_state_dict)
    model.load_state_dict(new_state_dict)
    print('not running forward pass')

  model.eval()
  loss, acc, ids = evaluate_data_loader(model, data_loader, device)
  print(f'alpha = {alpha}, interpolation loss {loss}, acc {acc}')
  return loss, acc, ids