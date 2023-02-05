
import abc
import argparse
from dataclasses import dataclass, fields
import copy

@dataclass
class Hparams(abc.ABC):
  """ collection of hyperparameters"""
  @classmethod
  def add_args(cls, parser, defaults: 'Hparams' = None, 
    name: str = None, description: str = None, create_group: bool = False):
    if defaults and not isinstance(defaults, cls):
      raise ValueError(f'defaults must also be type {cls}')
    
    for field in fields(cls):
      if field.name.startswith('_'): continue
      if defaults: default = copy.deepcopy(getattr(defaults, field.name, None))
      arg_name = f'--{field.name}'
      parser.add_argument(arg_name, type=field.type, default=default)
  
  @classmethod
  def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> 'Hparams':
    d = {}
    for field in fields(cls):
      arg_name = field.name
      if arg_name.startswith('_'): continue
      if not hasattr(args, arg_name): raise ValueError(f'Missing argument: {arg_name}')
      d[field.name] = getattr(args, arg_name)
    return cls(**d)
  
  @property
  def display(self):
    defined_fields = '\n'.join([f'     * {f.name} -> {getattr(self, f.name)}' for f in fields(self) if not f.name.startswith('_')])
    return self._name + '\n' + defined_fields

@dataclass
class TrainingHparams(Hparams):
  training_steps: str = '100ep'
  lr: float = 0.1
  momentum: float = 0.9
  milestone_steps: str = '50ep,75ep'
  weight_decay: float = 1e-4
  data_order_seed: int = None

  _name: str = 'Training Hyperparameters'

@dataclass
class DatasetHparams(Hparams):
  dataset_name: str = 'cifar10'
  batch_size: int = 128
  do_not_augment: bool = False
  transformation_seed: int = None
  subsample_fraction: float = None
  random_labels_fraction: str = None

  _name: str = 'Dataset Hyperparameters'

