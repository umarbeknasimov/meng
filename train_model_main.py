import os

import torch
import models
from training.hparams import MainArgs
from constants import USER_DIR, DEVICE
from training.train import train
from training.callbacks import standard_callbacks

# model: nn.Module, 
# args: HParams, 
# callbacks,
# output_location: str,
# start_step: Step = None, 
# end_step: Step = None

args = MainArgs()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

output_location = os.path.join(USER_DIR, 'new_framework')
model = models.frankleResnet20().to(DEVICE)
train(model, args, standard_callbacks(), output_location)