import torch
import copy
from utils import average
import evaluate

ITERATIONS = set([0] + [2**i for i in range(17)])
def train_epoch(args, train_loader, model, criterion, optimizer, epoch, steps_per_epoch, model_weights, device):
    """
        Run one train epoch
    """
    losses = average.AverageMeter()
    top1 = average.AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
      iteration = (epoch * steps_per_epoch) + i
      if iteration in ITERATIONS:
        model_weights.append(copy.deepcopy(model.state_dict()))
        print(f'saving weights on iteration {iteration}, len {len(model_weights)}')
        torch.save(model_weights, args.weights_filename)
      
      target = target.to(device)
      input_var = input.to(device)
      target_var = target

      # compute output
      output = model(input_var)
      loss = criterion(output, target_var)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      output = output.float()
      loss = loss.float()
      # measure accuracy and record loss
      prec1 = evaluate.accuracy(output.data, target)[0]
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))

      if i % args.print_freq == 0:
          print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    loss=losses, top1=top1))
            
    return top1.avg, losses.avg

def validate(args, val_loader, model, criterion, device):
    """
    Run evaluation
    """
    losses = average.AverageMeter()
    top1 = average.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = evaluate.accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

