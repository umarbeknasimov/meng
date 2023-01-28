import argparse
from dataclasses import dataclass
import os
import torch

import models
from foundations.runner import Runner
from foundations.hparams import TrainingHParams
from constants import USER_DIR, DEVICE
from training.train import train
from training.callbacks import standard_callbacks
import dataset

@dataclass
class TrainingRunner(Runner):
    trainingHParams: TrainingHParams
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return 'train a model'
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(TrainingHParams.create_from_args(args), args.verbose, args.evaluate_every_epoch)
    
    def run(self):
        if self.verbose:
            print('training a model')
            torch.manual_seed(self.trainingHParams.seed)
            torch.cuda.manual_seed(self.trainingHParams.seed)

            train_loader, test_loader = dataset.get_train_test_loaders()

            output_location = os.path.join(USER_DIR, 'new_framework', 'parent', 's_{}'.format(self.trainingHParams.seed))
            model = models.frankleResnet20().to(DEVICE)
            train(model, self.trainingHParams, standard_callbacks(self.trainingHParams, train_loader, test_loader), output_location, train_loader)

