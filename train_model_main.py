import os

import torch
import models
from foundations.hparams import TrainingHParams
from constants import USER_DIR, DEVICE
from training.train import train
from training.callbacks import standard_callbacks
import dataset

# model: nn.Module, 
# args: HParams, 
# callbacks,
# output_location: str,
# start_step: Step = None, 
# end_step: Step = None

args = TrainingHParams()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_loader, test_loader = dataset.get_train_test_loaders()

output_location = os.path.join(USER_DIR, 'new_framework')
model = models.frankleResnet20().to(DEVICE)
train(model, args, standard_callbacks(args, train_loader, test_loader), output_location, train_loader)