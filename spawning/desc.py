import argparse
import os

from dataclasses import dataclass
from cli.arg_utils import maybe_get_arg
from foundations import desc
from foundations import hparams
from foundations.hparams import ModelHparams, TrainingHparams, DatasetHparams
from foundations.step import Step
import datasets.registry
from foundations import desc
from environment import environment
from training.desc import TrainingDesc

@dataclass
class SpawningDesc(desc.Desc):
    training_hparams: TrainingHparams
    dataset_hparams: DatasetHparams
    model_hparams: ModelHparams
    pretrain_training_hparams: TrainingHparams = None
    pretrain_dataset_hparams: DatasetHparams = None

    @staticmethod
    def name_prefix(): return 'spawn'

    @staticmethod
    def _add_pretrain_argument(parser):
        parser.add_argument('--pretrain', action='store_true')

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: TrainingDesc = None) -> None:
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)

        pretrain = maybe_get_arg('pretrain', boolean_arg=True)
        SpawningDesc._add_pretrain_argument(parser)
        if pretrain:
            hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None, prefix='pretrain')
            hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None, prefix='pretrain')
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SpawningDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        desc = SpawningDesc(training_hparams, dataset_hparams, model_hparams)
        if args.pretrain:
            desc.pretrain_dataset_hparams = hparams.DatasetHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = hparams.TrainingHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name
        return desc

    def run_path(self, part='main', experiment='main'):
        path = os.path.join(
            environment.get_user_dir(), 
            experiment,
            self.hashname,
            part)
        return path
    
    def str_to_step(self, s: str, pretrain: bool = False) -> Step:
        dataset_hparams = self.pretrain_dataset_hparams if pretrain else self.dataset_hparams
        iterations_per_epoch = datasets.registry.get(dataset_hparams).iterations_per_epoch
        return Step.from_str(s, iterations_per_epoch)
    
    @property
    def pretrain_end_step(self):
        return self.str_to_step(self.pretrain_training_hparams.training_steps, True)
    
    @property
    def train_end_step(self):
        return self.str_to_step(self.training_hparams.training_steps)
    
    def _train_dataset_log2_steps(self):
        iterations_per_epoch = datasets.registry.get(self.dataset_hparams).iterations_per_epoch
        return Step.get_log_2_steps(self.train_end_step, iterations_per_epoch)
    
    @property
    def spawn_steps(self):
        return self._train_dataset_log2_steps()
    
    @property
    def children_saved_steps(self):
        return self._train_dataset_log2_steps()

    

    



