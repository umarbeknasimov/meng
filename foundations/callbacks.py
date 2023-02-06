import torch
import math

from datasets.base import DataLoader
from foundations import paths
from utils.evaluate import evaluate

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