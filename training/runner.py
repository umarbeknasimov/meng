from dataclasses import dataclass
import os

import models
from environment import environment
from foundations.runner import Runner
from foundations.step import Step
from training.train import train
from training.callbacks import standard_callbacks
from training.desc import TrainingDesc
from datasets import registry

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    verbose: bool = True

    @staticmethod
    def description():
        return 'train a model'
    
    def run(self):
        if self.verbose:
            print(f'running {self.description()}')
        output_location = self.training_desc.run_path()
        if not os.path.exists(output_location):
            os.makedirs(output_location)

        self.training_desc.save(output_location)
        train_loader = registry.get(self.training_desc.dataset_hparams)
        test_loader = registry.get(self.training_desc.dataset_hparams, False)

        model = models.frankleResnet20().to(environment.device())

        callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader)

        pretrained_output_location, pretrained_step = None, None
        if self.training_desc.pretrain_training_desc and self.training_desc.pretrain_step:
            pretrained_output_location = self.training_desc.pretrain_training_desc.run_path()
            pretrained_dataset = registry.get(self.training_desc.pretrain_training_desc.dataset_hparams)
            pretrained_step = Step.from_str(self.training_desc.pretrain_step, pretrained_dataset.iterations_per_epoch)

        train(
            model, 
            self.training_desc.training_hparams, 
            callbacks,
            output_location, 
            train_loader,
            pretrained_output_location,
            pretrained_step)


