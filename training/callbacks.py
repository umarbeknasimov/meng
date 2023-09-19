import torch.nn as nn
import numpy as np
import math

from datasets.base import DataLoader
from environment import environment
from foundations import paths
from foundations.callbacks import create_eval_callback, save_logger, save_model, save_optim
from foundations.step import Step
from foundations.hparams import ModelHparams, TrainingHparams
from models import cifar_resnet
import models.registry
from training import checkpointing
from utils import state_dict
from utils import interpolate
from utils.evaluate import evaluate
from utils.interpolate import interpolate_state_dicts_from_weights

#callback frequencies
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def run_every_x_iters_for_first_y_epochs(iters, epochs, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step.it % iters != 0 or step.ep > epochs:
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def run_every_step(callback):
    return callback

def run_at_step(target_step, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step != target_step:
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def run_at_steps(target_steps, callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step not in target_steps:
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def run_at_log_base_2_steps(callback):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step.iteration == 0:
            return
        log_base_2 = math.log2(step.iteration)
        if (math.ceil(log_base_2) != math.floor(log_base_2)):
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def run_at_every_x_steps(callback, x):
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step.iteration % x != 0:
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def run_at_log_base_2_steps_dense(callback, end_step: Step):
    # - Regular: 0, 1, 2, 4, 8, 16, 32, 64, ...
    # - Dense: 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, ...
    steps_in_iterations = set([step_i.iteration for step_i in Step.get_log_2_steps_dense(end_step)])
    def modified_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        if step.iteration not in steps_in_iterations:
            return
        callback(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    return modified_callback

def save_state_dicts(output_location, step, model, optimizer, scheduler, logger, ids_logger):
    save_optim(output_location, step, model, optimizer, scheduler, logger, ids_logger)
    save_model(output_location, step, model, optimizer, scheduler, logger, ids_logger)

def standard_callbacks(
    args: TrainingHparams,
    train_set_loader: DataLoader, 
    test_set_loader: DataLoader, 
    verbose: bool = True, 
    start_step: Step = None,
    evaluate_every_step: bool = False):
    
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    print('training steps ', args.training_steps)
    end = Step.from_str(args.training_steps, train_set_loader.iterations_per_epoch)

    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)


    result = [
        run_at_step(start, save_state_dicts),
        run_at_step(end, save_state_dicts),
        run_at_log_base_2_steps_dense(save_state_dicts, end),
        run_every_epoch(checkpointing.save_checkpoint_callback)
    ]

    result = result + [run_every_epoch(test_eval_callback), run_every_epoch(train_eval_callback)]
    if evaluate_every_step: result = result + [run_at_every_x_steps(train_eval_callback, 10), run_at_every_x_steps(test_eval_callback, 10)]
    
    result = result + [
        run_at_log_base_2_steps_dense(train_eval_callback, end),
        run_at_log_base_2_steps_dense(test_eval_callback, end),
        run_at_log_base_2_steps_dense(save_logger, end),
        run_every_epoch(save_logger),
        run_at_step(end, save_logger)]
    
    return result


def save_ema_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
    if step.iteration == 0:
        environment.save(model.state_dict(), paths.ema(output_location))
        return
    ema_weights = environment.load(paths.ema(output_location))
    curr_weights = model.state_dict()
    new_ema_weights = interpolate_state_dicts_from_weights(curr_weights, ema_weights, 0.99)
    print('saving ema')
    environment.save(new_ema_weights, paths.ema(output_location))

def create_warm_ema_callback(train_set_loader: DataLoader, model_hparams: ModelHparams):
    def warm_ema_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        ema_weights = environment.load(paths.ema(output_location))

        ema_model = models.registry.get(model_hparams).to(environment.device())
        averaged_weights_wo_batch_stats = state_dict.get_state_dict_wo_batch_stats(
            ema_model, ema_weights)
        ema_model.load_state_dict(averaged_weights_wo_batch_stats)
        interpolate.forward_pass(ema_model, train_set_loader)

        environment.save(ema_model.state_dict(), paths.ema_warm(output_location, step))
    return warm_ema_callback

def create_ema_eval_callback(eval_name: str, loader: DataLoader, model_hparams: ModelHparams):
    def eval_ema_callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
        ema_weights = environment.load(paths.ema_warm(output_location, step))
        ema_model = models.registry.get(model_hparams).to(environment.device())
        ema_model.load_state_dict(ema_weights)
        ema_model.eval()
        loss, accurary, _ = evaluate(ema_model, loader)

        logger.add('{}_loss'.format(eval_name), step, loss)
        logger.add('{}_accuracy'.format(eval_name), step, accurary)
        
        print('{}\tep: {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}'.format(
                eval_name, step.ep, step.it, loss, accurary))
    return eval_ema_callback

def standard_ema_callbacks(
    args: TrainingHparams,
    train_set_loader: DataLoader, 
    test_set_loader: DataLoader,  
    model_hparams: ModelHparams,
    start_step: Step = None):
    
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    print('training steps ', args.training_steps)
    end = Step.from_str(args.training_steps, train_set_loader.iterations_per_epoch)

    warm_ema_callback = create_warm_ema_callback(train_set_loader, model_hparams)

    test_ema_eval_callback = create_ema_eval_callback('test', test_set_loader, model_hparams)
    train_ema_eval_callback = create_ema_eval_callback('train', train_set_loader, model_hparams)


    result = [
        run_at_step(start, save_state_dicts),
        run_at_step(end, save_state_dicts),
        run_at_log_base_2_steps_dense(save_state_dicts, end),
        run_every_epoch(checkpointing.save_checkpoint_callback),
        save_ema_callback,
        run_at_log_base_2_steps_dense(warm_ema_callback, end),
        run_at_step(start, warm_ema_callback),
        run_at_step(end, warm_ema_callback),
    ]
    
    result = result + [
        run_at_log_base_2_steps_dense(test_ema_eval_callback, end),
        run_at_log_base_2_steps_dense(train_ema_eval_callback, end),
        run_at_log_base_2_steps_dense(save_logger, end),
        run_every_epoch(save_logger),
        run_at_step(end, save_logger)]
    
    return result

    

