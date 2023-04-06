import argparse
from dataclasses import dataclass
import datasets.registry

from cli import shared_args
from environment import environment
from foundations.runner import Runner
import models.registry
from training.callbacks import standard_callbacks
from training.train import standard_train
from training.desc import TrainingDesc

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    experiment: str = 'main'
    save_dense: bool = False
    

    @staticmethod
    def description():
        return 'train a model'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(TrainingDesc.create_from_args(args), experiment=args.experiment, save_dense=args.save_dense)
    
    def run(self):
        print(f'running {self.description()}')

        # train_loader = datasets.registry.get(self.training_desc.dataset_hparams)
        # test_loader = datasets.registry.get(self.training_desc.dataset_hparams, False)
        # callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader, save_dense=self.save_dense)

        model = models.registry.get(self.training_desc.model_hparams).to(environment.device())

        output_location = self.training_desc.run_path(self.experiment)
        environment.exists_or_makedirs(output_location)
        self.training_desc.save_hparam(output_location)

        # train(model, self.training_desc.training_hparams, train_loader, output_location, callbacks)

        standard_train(
                model, output_location, self.training_desc.dataset_hparams, 
                self.training_desc.training_hparams,
                save_dense=self.save_dense)