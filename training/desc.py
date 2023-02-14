import argparse
from dataclasses import dataclass
import os


from environment import environment
from foundations import desc, hparams, paths
from foundations.step import Step

@dataclass
class TrainingDesc(desc.Desc):
    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams

    @staticmethod
    def name_prefix(): return 'train'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'TrainingDesc' = None):
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        return TrainingDesc(model_hparams, dataset_hparams, training_hparams)

    def run_path(self, experiment='main'):
        return paths.train(os.path.join(
            environment.get_user_dir(),
            experiment, 
            self.hashname))
