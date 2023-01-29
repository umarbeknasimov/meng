from dataclasses import dataclass
import os
import torch

import models
from environment import environment
from foundations.runner import Runner
from training.train import train
from training.callbacks import standard_callbacks
from training.desc import TrainingDesc
import dataset

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    verbose: bool = True

    @staticmethod
    def description():
        return 'train a model'
    
    def run(self):
        if self.verbose:
            print('training a model')
        output_path = self.training_desc.run_path()
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.training_desc.save(output_path)

        torch.manual_seed(self.training_desc.training_hparams.seed)
        torch.cuda.manual_seed(self.training_desc.training_hparams.seed)

        train_loader, test_loader = dataset.get_train_test_loaders()
        model = models.frankleResnet20().to(environment.device())

        if self.training_desc.init_state_dict_path():
            init_state_dict = torch.load(self.training_desc.init_state_dict_path())
            model.load_state_dict(init_state_dict['model'])
            train(model, self.training_desc.training_hparams, standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader), output_path, train_loader, init_state_dict['optimizer'], init_state_dict['scheduler'])
        train(model, self.training_desc.training_hparams, standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader), output_path, train_loader)

