from environment import environment
from datasets.base import DataLoader
from foundations.step import Step
from foundations import paths
from training.metric_logger import MetricLogger
from training import pre_trained
from utils import state_dict, interpolate

def average(
    model_type,
    output_location_1: str,
    output_location_2: str,
    output_location: str,
    steps: list[Step],
    train_loader: DataLoader,
    callbacks
    ):

    logger = MetricLogger()

    for step in steps:
        model = model_type().to(environment.device())
        weights1 = pre_trained.get_pretrained_model(output_location_1, step)
        weights2 = pre_trained.get_pretrained_model(output_location_2, step)
        new_state_dict = interpolate.interpolate_state_dicts_from_weights(weights1, weights2, 0.5)
        new_state_dict = state_dict.get_state_dict_w_o_batch_stats(new_state_dict)
        model.load_state_dict(new_state_dict)
        interpolate.forward_pass(model, train_loader)

        for callback in callbacks: callback(output_location, step, model, None, None, logger)



