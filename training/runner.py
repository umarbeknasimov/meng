import argparse
from dataclasses import dataclass
import datasets.registry

from cli import shared_args
from environment import environment
from foundations.runner import Runner
import models.registry
from training import train
from training.callbacks import standard_callbacks
from training.train import standard_train
from training.desc import TrainingDesc

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    save_dense: bool = False
    experiment: str = 'main'
    eval_every_10: bool = False
    

    @staticmethod
    def description():
        return 'train a model'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        parser.add_argument('--eval_every_10', action='store_true')
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(TrainingDesc.create_from_args(args), experiment=args.experiment, eval_every_10=args.eval_every_10)
    
    def run(self):
        print(f'running {self.description()}')
        
        outputs = datasets.registry.get(self.training_desc.dataset_hparams).dataset.num_classes()
        model = models.registry.get(self.training_desc.model_hparams, outputs).to(environment.device())

        output_location = self.training_desc.run_path(self.experiment)
        environment.exists_or_makedirs(output_location)
        self.training_desc.save_hparam(output_location)

        standard_train(
                model, output_location, self.training_desc.dataset_hparams, 
                self.training_desc.training_hparams, evaluate_every_10=self.eval_every_10)