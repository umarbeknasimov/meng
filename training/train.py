import torch.nn as nn
from torch.utils.data import DataLoader

from environment import environment
from foundations.hparams import TrainingHparams
from foundations.step import Step
from training.metric_logger import MetricLogger
from training import optimizers

def train(
    model: nn.Module, 
    args: TrainingHparams, 
    callbacks,
    output_location: str,
    train_loader: DataLoader,
    init_optimizer_state = None,
    init_lr_scheduler_state = None,
    start_step: Step = None, 
    end_step: Step = None):

    logger = MetricLogger()
    
    iterations_per_epoch = len(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.get_optimizer(model, args, init_optimizer_state)

    start_step = start_step or Step.zero(iterations_per_epoch)
    end_step = end_step or Step.from_str(args.training_steps, iterations_per_epoch)
    
    lr_scheduler = optimizers.get_lr_scheduler(args, iterations_per_epoch, optimizer, init_lr_scheduler_state)
    
    if start_step > end_step:
        return
    for ep in range(start_step.ep, end_step.ep):
        for it, (input, target) in enumerate(train_loader):

            # advance dataloader until start epoch and iteration
            if ep == start_step.ep and it < start_step.it: continue

            if ep == end_step.ep and it == end_step.it: return

            step = Step.from_epoch(ep, it, iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, lr_scheduler, logger)
            
            target = target.to(environment.device())
            input_var = input.to(environment.device())

            # compute output
            output = model(input_var)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()