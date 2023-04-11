from dataclasses import dataclass
from foundations.hparams import Hparams
import models.registry
from cli import arg_utils

@dataclass
class JobArgs(Hparams):
    """Arguments shared across lottery ticket jobs."""

    replicate: int = 1
    default_hparams: str = None
    quiet: bool = False
    experiment: str = 'main'

    _name: str = 'High-Level Arguments'
    _description: str = 'Arguments that determine how the job is run and where it is stored.'
    _replicate: str = 'The index of this particular replicate. ' \
                      'Use a different replicate number to run another copy of the same experiment.'
    _default_hparams: str = 'Populate all arguments with the default hyperparameters for this model.'
    _quiet: str = 'Suppress output logging about the training status.'
    _experiment: str = 'Experiment name that will be used to store results.'

def maybe_get_default_hparams():
    default_hparams = arg_utils.maybe_get_arg('default_hparams')
    return models.registry.get_default_hparams(default_hparams) if default_hparams else None