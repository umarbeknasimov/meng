from datasets.base import DataLoader
import datasets.registry
from environment import environment
from foundations import paths
from foundations.hparams import DatasetHparams, ModelHparams
from foundations.step import Step
from foundations.callbacks import create_eval_callback, save_logger, save_model
import models.registry
from training.metric_logger import MetricLogger
from utils import state_dict, interpolate
from models.registry import get_model_state_dict

def average(
    model_hparams: ModelHparams,
    output_location: str,
    models_location: str,
    callbacks: list,
    train_loader: DataLoader,
    seeds: list,
    step: Step
    ):
    """average models at some step across multiple seeds"""
    environment.exists_or_makedirs(output_location)
    if environment.exists(paths.logger(output_location)):
        logger = MetricLogger.create_from_file(output_location)
    else:
        logger = MetricLogger()
    weights = []
    for data_order_seed in seeds:
        weights.append(
            get_model_state_dict(
                paths.seed(models_location, data_order_seed),
                step))
    model = models.registry.get(model_hparams).to(environment.device())
    averaged_weights = interpolate.average_state_dicts(weights)
    averaged_weights_wo_batch_stats = state_dict.get_state_dict_wo_batch_stats(
        model, averaged_weights)
    model.load_state_dict(averaged_weights_wo_batch_stats)
    interpolate.forward_pass(model, train_loader)
    for callback in callbacks: callback(
        output_location,
        step,
        model,
        None,
        None,
        logger)

def standard_average(
    dataset_hparams: DatasetHparams,
    model_hparams: ModelHparams,
    output_location: str,
    models_location: str,
    seeds: list,
    step: Step
):
    train_loader = datasets.registry.get(dataset_hparams)
    test_loader = datasets.registry.get(dataset_hparams, False)
    test_eval_callback = create_eval_callback('test', test_loader)
    train_eval_callback = create_eval_callback('train', train_loader)
    callbacks = [save_model, test_eval_callback, train_eval_callback, save_logger]

    return average(model_hparams, output_location, models_location, callbacks, train_loader, seeds, step)

