import argparse
from dataclasses import dataclass
import datasets.registry

from cli import shared_args
from environment import environment
from foundations.runner import Runner
import models.registry
from training.callbacks import run_every_x_iters_for_first_y_epochs, save_state_dicts, standard_callbacks
from training.train import train
from training.desc import TrainingDesc

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    experiment: str = 'main'
    

    @staticmethod
    def description():
        return 'train a model'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(TrainingDesc.create_from_args(args), experiment=args.experiment)
    
    def run(self):
        print(f'running {self.description()}')

        train_loader = datasets.registry.get(self.training_desc.dataset_hparams)
        test_loader = datasets.registry.get(self.training_desc.dataset_hparams, False)
        callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader)
        callbacks = [run_every_x_iters_for_first_y_epochs(5, 10, save_state_dicts)] + callbacks

        model = models.registry.get(self.training_desc.model_hparams).to(environment.device())

        output_location = self.training_desc.run_path(self.experiment)
        environment.exists_or_makedirs(output_location)
        self.training_desc.save_hparam(output_location)

        train(model, self.training_desc.training_hparams, train_loader, output_location, callbacks)