from dataclasses import dataclass
import os
import math

import models
from environment import environment
from foundations.runner import Runner
from foundations.step import Step
from training.train import train
from training.callbacks import standard_callbacks
from averaging.desc import AveragingDesc
from averaging.average import average
from datasets import registry

@dataclass
class AveragingRunner(Runner):
    average_desc: AveragingDesc
    verbose: bool = True

    @staticmethod
    def description():
        return 'average 2 models and evaluate'
    
    def run(self):
        if self.verbose:
            print(f'running {self.description()}')
        output_location = self.training_desc.run_path()
        output_location_outer = self.training_desc.run_path(get_outer_dir=True)
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        
        if not os.path.exists(output_location_outer):
            os.makedirs(output_location_outer)

        self.training_desc.save(output_location, output_location_outer)
        train_loader = registry.get(self.average_desc.dataset_hparams)
        test_loader = registry.get(self.average_desc.dataset_hparams, False)


        callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader)

        train1 = AveragingDesc.train1
        train2 = AveragingDesc.train2

        if train1.training_hparams.training_steps != train2.training_hparams.training_steps: 
            raise ValueError(f'{train1.training_hparams.training_steps} training steps for train1 does not equal {train2.training_hparams.training_steps} training steps for train2')
        
        train1_train2_dataset = train1.dataset_hparams
        iterations_per_epoch = registry.get(train1_train2_dataset).iterations_per_epoch
        end_step = Step.from_str(train1_train2_dataset.training_steps, iterations_per_epoch)
        steps = Step.get_log_2_steps(end_step, iterations_per_epoch)
        average(models.frankleResnet20, train1.run_path(), train2.run_path(), self.average_desc.run_path(), steps, train_loader, callbacks)

        

        


