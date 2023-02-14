
import abc
import argparse
from dataclasses import MISSING, dataclass, fields, asdict
import copy

@dataclass
class Hparams(abc.ABC):
  """ collection of hyperparameters"""
  @classmethod
  def add_args(cls, parser, defaults: 'Hparams' = None, prefix: str = None):
    if defaults and not isinstance(defaults, cls):
      raise ValueError(f'defaults must also be type {cls}')
    
    for field in fields(cls):
      if field.name.startswith('_'): continue
      if defaults: default = copy.deepcopy(getattr(defaults, field.name, None))
      elif field.default != MISSING: default = copy.deepcopy(field.default)
      else: default = None
      arg_name = f'--{field.name}' if prefix is None else f'--{prefix}_{field.name}'
      if field.type == bool:
        if (default and getattr(defaults, field.name) is not False or field.default is not False):
          raise ValueError(f'bool hyperparams must default to false: {field.name}')
        parser.add_argument(arg_name, action='store_true') 
      elif field.type in [str, float, int]:
        required = field.default is MISSING and default is None
        parser.add_argument(arg_name, type=field.type, default=default, required=required)
      # parser.add_argument(arg_name, type=field.type, default=default)
  
  @classmethod
  def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> 'Hparams':
    d = {}
    print('args ', args)
    for field in fields(cls):
      if field.name.startswith('_'): continue
      arg_name = f'{field.name}' if prefix is None else f'{prefix}_{field.name}'
      if not hasattr(args, arg_name): raise ValueError(f'Missing argument: {arg_name}')
      d[field.name] = getattr(args, arg_name)
    return cls(**d)
  
  @classmethod
  def create_from_dict(cls, args: dict, prefix: str = None) -> 'Hparams':
    d = {}
    for field in fields(cls):
      arg_name = field.name
      if arg_name.startswith('_'): continue
      if arg_name not in args: raise ValueError(f'Missing argument: {arg_name}')
      d[field.name] = args[arg_name]
    return cls(**d)
  
  @staticmethod
  def create_from_instance_and_dict(instance: 'Hparams', args: dict, prefix: str = None) -> 'Hparams':
    instance_dict = asdict(instance)
    for field_name in args:
      if field_name not in instance_dict:
        raise ValueError(f'dict contains incorrect field name {field_name}')
      instance_dict[field_name] = args[field_name]
    return instance.create_from_dict(instance_dict, prefix)
      
  
  @property
  def display(self):
    defined_fields = '\n'.join([f'     * {f.name} -> {getattr(self, f.name)}' for f in fields(self) if not f.name.startswith('_')])
    return self._name + '\n' + defined_fields

@dataclass
class TrainingHparams(Hparams):
  training_steps: str = '160ep'
  lr: float = 0.1
  gamma: float = 0.1
  momentum: float = 0.9
  milestone_steps: str = '80ep,120ep'
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
  random_labels_fraction: float = None

  _name: str = 'Dataset Hyperparameters'

@dataclass
class ModelHparams(Hparams):
  model_name: str = 'cifar_resnet_20'

  _name: str = 'Model Hyperparameters'
