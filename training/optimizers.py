import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from foundations.hparams import TrainingHparams
from foundations.step import Step
from . import lookahead_optim

def get_optimizer(model: nn.Module, args: TrainingHparams) -> Optimizer:
    if args.optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif args.optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif args.optimizer_name == 'lookahead':
        optim = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        return lookahead_optim.Lookahead(optim)
    
    raise ValueError(f'no such optimizer {args.optimizer_name}')

def get_lr_scheduler(args: TrainingHparams, iterations_per_epoch: int, optimizer: Optimizer) -> MultiStepLR:
    if args.milestone_steps is None:
        return None
    lr_milestones = [Step.from_str(x, iterations_per_epoch).iteration for x in args.milestone_steps.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=lr_milestones,
                                                            gamma=args.gamma)
    return lr_scheduler