import os

import models
from child_spawn.desc import ChildSpawnDesc
from environment import environment
from foundations.runner import Runner
from foundations.step import Step
from training.train import train
from training.callbacks import standard_callbacks

from datasets import registry


class ChildSpawnRunner(Runner):
    child_spawn_desc = ChildSpawnDesc
    verbose: bool = True

    @staticmethod
    def description():
        return 'spawn and train children from a model run'
    
    def run(self, child_index):
        if self.verbose:
            print(f'running {self.description()} for index {child_index}')
        output_location = self.child_spawn_desc.run_path()
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        
        parent_desc = self.child_spawn_desc.parent_training_desc
        child_desc = self.child_spawn_desc.parent_training_desc

        train_loader = registry.get(child_desc.dataset_hparams)
        test_loader = registry.get(child_desc.dataset_hparams, False)

        model = models.frankleResnet20().to(environment.device())

        callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader)

        pretrained_output_location = parent_desc.run_path()
        pretrained_dataset = registry.get(parent_desc.dataset_hparams)
        pretrained_step = Step.from_str(parent_desc.pretrain_step, pretrained_dataset.iterations_per_epoch)

        train(
            model, 
            child_desc, 
            callbacks,
            output_location, 
            train_loader,
            pretrained_output_location,
            pretrained_step)
        


