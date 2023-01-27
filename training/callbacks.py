import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.average import AverageMeter
from foundations.step import Step
from foundations import paths
from foundations.hparams import TrainingHParams
from constants import EXPONENTIAL_STEPS, DEVICE
import evaluate

def save_state_dicts(output_location, step, model, optimizer, scheduler, logger):
    torch.save(paths.state_dict('model', output_location, step), model.state_dict())
    torch.save(paths.state_dict('optimizer', output_location, step), optimizer.state_dict())
    torch.save(paths.state_dict('scheduler', output_location, step), scheduler.state_dict())

def save_logger(output_location, step, model, optimizer, scheduler, logger):
    logger.save(output_location)

def create_eval_callback(eval_name: str, loader: DataLoader, verbose=True):
    def eval_callback(output_location, step, model, optimizer, scheduler, logger):
        criterion = nn.CrossEntropyLoss()
        losses = AverageMeter()
        top1 = AverageMeter()
       
        with torch.no_grad():
            for i, (input, target) in enumerate(loader):                
                target = target.to(DEVICE)
                input_var = input.to(DEVICE)

                # compute output
                output = model(input_var)
                loss = criterion(output, target)

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                prec1 = evaluate.accuracy(output.data, target)[0]
                losses.update(loss.cpu().item(), input.cpu().size(0))
                top1.update(prec1.cpu().item(), input.cpu().size(0))

            logger.add('{}_loss'.format(eval_name), step, losses.avg)
            logger.add('{}_accuracy'.format(eval_name), step, top1.avg)
        
        if verbose:
            print('{}\tep: {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}'.format(
                    eval_name, step.ep, step.it, losses.avg, top1.avg))
    return eval_callback

#callback frequencies
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, scheduler, logger)
    return modified_callback

def run_every_step(callback):
    return callback

def run_at_step(target_step, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step != target_step:
            return
        callback()
    return modified_callback

def run_at_steps(target_steps, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step not in target_steps:
            return
        callback()
    return modified_callback

def standard_callbacks(
    args: TrainingHParams,
    train_set_loader: DataLoader, 
    test_set_loader: DataLoader, 
    eval_on_train: bool = True, 
    verbose: bool = True, 
    start_step: Step = None,
    evaluate_every_epoch: bool = True):
    
    iterations_per_epoch = len(train_set_loader)
    start = start_step or Step.zero(iterations_per_epoch)
    end = Step.from_str(args.training_steps, iterations_per_epoch)

    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    result = [
        run_at_step(start, save_state_dicts),
        run_at_step(end, save_state_dicts),
        run_every_epoch(save_logger),
    ]

    if evaluate_every_epoch: result = [run_every_epoch(test_eval_callback)] + result
    if eval_on_train:
        if evaluate_every_epoch: result = [run_every_epoch(train_eval_callback)] + result
    
    result = [run_at_steps(EXPONENTIAL_STEPS, save_state_dicts)] + result
    
    return result

    

