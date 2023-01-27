import torch
import torch.nn as nn
import copy

import dataset
from foundations.hparams import TrainingHParams
from foundations.step import Step
from constants import DEVICE
from training.metric_logger import MetricLogger

def train(
    model: nn.Module, 
    args: TrainingHParams, 
    callbacks,
    output_location: str,
    start_step: Step = None, 
    end_step: Step = None):

    logger = MetricLogger()
    train_loader, _ = dataset.get_train_val_loaders()
    
    iterations_per_epoch = len(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    start_step = start_step or Step.zero(iterations_per_epoch)
    end_step = end_step or Step.from_str(args.training_steps)
    
    # lr_milestones is in # of iterations
    lr_milestones = [Step.from_str(x, iterations_per_epoch).iteration for x in args.milestone_steps.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=lr_milestones, last_epoch=start_step.iteration - 1)
    
    if start_step > end_step:
        return
    for ep in range(start_step.ep, end_step.ep):
        for it, (input, target) in enumerate(train_loader):

            # advance dataloader until start epoch and iteration
            if ep == start_step.ep and it < start_step.it: continue

            if ep == end_step.ep and it == end_step.it: return

            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, lr_scheduler, logger)
            
            target = target.to(DEVICE)
            input_var = input.to(DEVICE)

            # compute output
            output = model(input_var)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()