import torch
import torch.nn as nn
from datasets.base import DataLoader
from utils.average import AverageMeter
from environment import environment

def evaluate(model, loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()
    correct_ids = []
    ids_multiplier_offset = int(loader.batch_size)
    
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):                
            target = target.to(environment.device())
            input_var = input.to(environment.device())

            # compute output
            output = model(input_var)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.cpu().item(), input.cpu().size(0))
            top1.update(prec1.cpu().item(), input.cpu().size(0))

            curr_ids_correct = ((output.data.argmax(-1) == target) == True).nonzero()
            if (len(curr_ids_correct.shape) == 2):
                curr_ids_correct = curr_ids_correct.squeeze(dim=1)
            # print(((torch.ones(curr_ids_correct.shape) * i * ids_multiplier_offset)))
            correct_ids.extend((curr_ids_correct + ((torch.ones(curr_ids_correct.shape) * i * ids_multiplier_offset))).int().tolist())
    print(correct_ids)
    return losses.avg, top1.avg, correct_ids

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