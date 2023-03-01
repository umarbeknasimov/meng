import torch.nn as nn
import math

from datasets.base import DataLoader
from foundations.callbacks import create_eval_callback, save_logger, save_model, save_optim
from foundations.step import Step
from foundations.hparams import TrainingHparams
from training import checkpointing

#callback frequencies
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, scheduler, logger)
    return modified_callback

def run_every_x_iters_for_first_y_epochs(iters, epochs, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step.it % iters != 0 or step.ep > epochs:
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

def run_at_log_base_2_steps_dense(callback):
    # - Regular: 0, 1, 2, 4, 8, 16, 32, 64, ...
    # - Dense: 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, ...
    def modified_callback(output_location, step, model, optimizer, scheduler, logger):
        if step.iteration == 0:
            return
        log_base_2 = math.log2(step.iteration)
        log_base_2_dense = math.log2(step.iteration/3)
        if (math.ceil(log_base_2) != math.floor(log_base_2)) and (math.ceil(log_base_2_dense) != math.floor(log_base_2_dense)):
            return
        callback(output_location, step, model, optimizer, scheduler, logger)
    return modified_callback

def save_state_dicts(output_location, step, model, optimizer, scheduler, logger):
    save_optim(output_location, step, model, optimizer, scheduler, logger)
    save_model(output_location, step, model, optimizer, scheduler, logger)

def standard_callbacks(
    args: TrainingHparams,
    train_set_loader: DataLoader, 
    test_set_loader: DataLoader, 
    eval_on_train: bool = True, 
    verbose: bool = True, 
    start_step: Step = None,
    evaluate_every_epoch: bool = True,
    save_dense: bool = False):
    
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    print('training steps ', args.training_steps)
    end = Step.from_str(args.training_steps, train_set_loader.iterations_per_epoch)

    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    if save_dense:
        run_at_log_base_2 = run_at_log_base_2_steps_dense
    else:
        run_at_log_base_2 = run_at_log_base_2_steps


    result = [
        run_at_step(start, save_state_dicts),
        run_at_step(end, save_state_dicts),
        run_at_log_base_2(save_state_dicts),
        run_every_epoch(checkpointing.save_checkpoint_callback)
    ]

    if evaluate_every_epoch: result = result + [run_every_epoch(test_eval_callback)]
    if eval_on_train:
        if evaluate_every_epoch: result = result + [run_every_epoch(train_eval_callback)]
    
    result = result + [
        run_at_log_base_2(train_eval_callback),
        run_at_log_base_2(test_eval_callback),
        run_at_log_base_2(save_logger),
        run_every_epoch(save_logger),
        run_at_step(end, save_logger)]
    
    return result

    

