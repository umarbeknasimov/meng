import argparse
from dataclasses import dataclass
import os
import datasets.registry
from environment import environment
from foundations import desc
from foundations import hparams
from foundations.hparams import DatasetHparams, ModelHparams, TrainingHparams
from foundations.step import Step
from training.desc import TrainingDesc


@dataclass
class SplitMergeDesc(desc.Desc):
    training_hparams: TrainingHparams
    dataset_hparams: DatasetHparams
    model_hparams: ModelHparams
    strategy: str = None

    @staticmethod
    def name_prefix(): return 'split_merge'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: TrainingDesc = None) -> None:
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        parser.add_argument('--strategy', type=str, default=None)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SplitMergeDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        return SplitMergeDesc(training_hparams, dataset_hparams, model_hparams, args.strategy)
        
    def run_path(self, part='main', experiment='main'):
        path = os.path.join(
            environment.get_user_dir(), 
            experiment,
            self.hashname,
            part)
        return path
    
    @property
    def train_end_step(self):
        return Step.from_str(
            self.training_hparams.training_steps, 
            datasets.registry.get(self.dataset_hparams).iterations_per_epoch)
