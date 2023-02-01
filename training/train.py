import torch
import torch.nn as nn

from environment import environment
from foundations.hparams import TrainingHparams
from foundations.step import Step
from datasets.base import DataLoader
from training.metric_logger import MetricLogger
from training import optimizers
from training.pre_trained import load_pretrained

def train(
    model: nn.Module, 
    training_hparams: TrainingHparams, 
    callbacks,
    output_location: str,
    train_loader: DataLoader,
    pretrained_output_location: str = None,
    pretrained_step: Step = None,
    start_step: Step = None, 
    end_step: Step = None):

    logger = MetricLogger()

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.get_optimizer(model, training_hparams)

    start_step = start_step or Step.zero(train_loader.iterations_per_epoch)
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)

    data_order_seed = training_hparams.data_order_seed
    if data_order_seed is not None:
        data_order_seed_generator = torch.Generator()
        data_order_seed_generator.manual_seed(data_order_seed)
        data_order_seed = torch.randint(int(1e8), (1,), generator=data_order_seed_generator).item()
    
    scheduler = optimizers.get_lr_scheduler(training_hparams, train_loader.iterations_per_epoch, optimizer)
    if pretrained_output_location and pretrained_step: load_pretrained(pretrained_output_location, pretrained_step, model, optimizer, scheduler)
    
    if start_step > end_step:
        return
    for ep in range(start_step.ep, end_step.ep):
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))
        for it, (input, target) in enumerate(train_loader):

            # advance dataloader until start epoch and iteration
            if ep == start_step.ep and it < start_step.it: continue

            if ep == end_step.ep and it == end_step.it: return

            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, scheduler, logger)
            
            target = target.to(environment.device())
            input_var = input.to(environment.device())

            # compute output
            output = model(input_var)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()