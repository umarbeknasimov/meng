import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from foundations.hparams import TrainingHParams
from foundations.step import Step

def get_optimizer(model: nn.Module, args: TrainingHParams, init_optimizer_state = None) -> Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    if init_optimizer_state:
        optimizer.load_state_dict(init_optimizer_state)
    return optimizer

def get_lr_scheduler(args: TrainingHParams, iterations_per_epoch: int, optimizer: Optimizer, init_scheduler_state = None) -> MultiStepLR:
    lr_milestones = [Step.from_str(x, iterations_per_epoch).iteration for x in args.milestone_steps.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=lr_milestones)
    
    if init_scheduler_state:
        lr_scheduler.load_state_dict(init_scheduler_state)
    return lr_scheduler