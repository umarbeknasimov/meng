import torch
import torch.nn as nn
import math

from environment import environment
from datasets.base import DataLoader
from utils.average import AverageMeter
from foundations.step import Step
from foundations import paths
from foundations.hparams import TrainingHparams
from utils.evaluate import evaluate, accuracy
from training import checkpointing

def save_state_dicts(output_location, step, model, optimizer, scheduler, logger):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, paths.state_dict(output_location, step))

def save_model(output_location, step, model, optimizer, scheduler, logger):
    torch.save(model.state_dict(), paths.model(output_location, step))

def save_logger(output_location, step, model, optimizer, scheduler, logger):
    logger.save(output_location)

def create_eval_callback(eval_name: str, loader: DataLoader, verbose=True):
    def eval_callback(output_location, step, model, optimizer, scheduler, logger):
        loss, accurary = evaluate(model, loader)

        logger.add('{}_loss'.format(eval_name), step, loss)
        logger.add('{}_accuracy'.format(eval_name), step, accurary)
        
        if verbose:
            print('{}\tep: {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}'.format(
                    eval_name, step.ep, step.it, loss, accurary))
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
        callback(output_location, step, model, optimizer, scheduler, logger)
    return modified_callback

def run_at_steps(target_steps, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step not in target_steps:
            return
        callback(output_location, step, model, optimizer, scheduler, logger)
    return modified_callback

def run_at_log_base_2_steps(callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step.iteration == 0:
            return
        log_base_2 = math.log2(step.iteration)
        if (math.ceil(log_base_2) != math.floor(log_base_2)):
            return
        callback(output_location, step, model, optimizer, scheduler, logger)
    return modified_callback

def standard_callbacks(
    args: TrainingHparams,
    train_set_loader: DataLoader, 
    test_set_loader: DataLoader, 
    eval_on_train: bool = True, 
    verbose: bool = True, 
    start_step: Step = None,
    evaluate_every_epoch: bool = True):
    
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    print('training steps ', args.training_steps)
    end = Step.from_str(args.training_steps, train_set_loader.iterations_per_epoch)

    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    result = [
        run_at_step(start, save_state_dicts),
        run_at_step(end, save_state_dicts),
        run_at_step(end, save_logger),
        run_every_epoch(save_logger),
        run_every_epoch(checkpointing.save_checkpoint_callback)
    ]

    if evaluate_every_epoch: result = [run_every_epoch(test_eval_callback)] + result
    if eval_on_train:
        if evaluate_every_epoch: result = [run_every_epoch(train_eval_callback)] + result
    
    result = [
        run_at_log_base_2_steps(save_state_dicts), 
        run_at_log_base_2_steps(train_eval_callback),
        run_at_log_base_2_steps(test_eval_callback),
        run_at_log_base_2_steps(save_logger)] + result
    
    return result

    

