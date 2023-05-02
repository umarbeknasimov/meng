from datasets.base import DataLoader
import datasets.registry
from environment import environment
from foundations import paths
from foundations.hparams import DatasetHparams, ModelHparams, TrainingHparams
from foundations.step import Step
from foundations.callbacks import create_eval_callback, save_logger, save_model
import models.registry
from training import optimizers
from training.callbacks import save_state_dicts
from training.metric_logger import MetricLogger
from utils import state_dict, interpolate
from models.registry import get_model_state_dict, get_optim_state_dict

def average(
    model_hparams: ModelHparams,
    training_hparams: TrainingHparams,
    output_location: str,
    callbacks: list,
    train_loader: DataLoader,
    step: Step,
    weights,
    optimizer_weights
    ):
    """average models at some step across multiple seeds"""
    environment.exists_or_makedirs(output_location)
    if environment.exists(paths.logger(output_location)):
        logger = MetricLogger.create_from_file(output_location)
    else:
        logger = MetricLogger()

    model = models.registry.get(model_hparams, train_loader.dataset.num_classes()).to(environment.device())
    averaged_weights = interpolate.average_state_dicts(weights)
    averaged_weights_wo_batch_stats = state_dict.get_state_dict_wo_batch_stats(
        model, averaged_weights)
    model.load_state_dict(averaged_weights_wo_batch_stats)
    interpolate.forward_pass(model, train_loader)

    averaged_optimizer_weights = interpolate.average_optimizer_state_dicts(optimizer_weights)
    optimizer = optimizers.get_optimizer(model, training_hparams)
    optimizer.load_state_dict(averaged_optimizer_weights)

    for callback in callbacks: callback(
        output_location,
        step,
        model,
        optimizer,
        None,
        logger)

def standard_average(
    dataset_hparams: DatasetHparams,
    model_hparams: ModelHparams,
    training_hparams: TrainingHparams,
    output_location: str,
    step: Step,
    weights,
    optimizer_weights,
    dont_save_models = False
):
    train_loader = datasets.registry.get(dataset_hparams)
    test_loader = datasets.registry.get(dataset_hparams, False)
    test_eval_callback = create_eval_callback('test', test_loader)
    train_eval_callback = create_eval_callback('train', train_loader)
    callbacks = [test_eval_callback, train_eval_callback, save_logger]
    if not dont_save_models:
        callbacks = [save_state_dicts] + callbacks

    return average(model_hparams, training_hparams, output_location, callbacks, train_loader, step, weights, optimizer_weights)

