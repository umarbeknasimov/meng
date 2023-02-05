import torch
import torch.nn as nn

from environment import environment
from foundations.hparams import TrainingHparams, DatasetHparams
from foundations.step import Step
from datasets.base import DataLoader
import datasets
from training.callbacks import standard_callbacks
from training.metric_logger import MetricLogger
from training import optimizers
from training.pre_trained import load_pretrained

def train( 
    model: nn.Module,
    training_hparams: TrainingHparams,
    train_loader: DataLoader,
    output_location: str,
    callbacks,
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
    for ep in range(start_step.ep, end_step.ep + 1):
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

def standard_train(
  model: nn.Module,
  output_location: str,
  dataset_hparams: DatasetHparams,
  training_hparams: TrainingHparams,
  pretrained_output_location: str = None,
  pretrained_step: Step = None,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True
):
    """Train using the standard callbacks according to the provided hparams."""

    # # If the model file for the end of training already exists in this location, do not train.
    # iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    # train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    train(model, training_hparams, train_loader, output_location, callbacks, pretrained_output_location, pretrained_step, start_step=start_step)