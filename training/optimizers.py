import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from foundations.hparams import TrainingHparams
from foundations.step import Step

def get_optimizer(model: nn.Module, args: TrainingHparams) -> Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    return optimizer

def get_lr_scheduler(args: TrainingHparams, iterations_per_epoch: int, optimizer: Optimizer) -> MultiStepLR:
    lr_milestones = [Step.from_str(x, iterations_per_epoch).iteration for x in args.milestone_steps.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=lr_milestones)
    return lr_scheduler