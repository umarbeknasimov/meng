import torch
import torch.nn as nn
import copy

import dataset
import train
import models

def main(model, args, device):
  iteration_model_weights = []

  train_loader, val_loader = dataset.get_train_val_loaders()
  
  steps_per_epoch = len(train_loader)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=[33, 66], last_epoch=args.start_epoch - 1)
  for epoch in range(args.start_epoch, args.epochs):

    # train for one epoch
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train.train_epoch(args, train_loader, model, criterion, 
                                  optimizer, epoch, steps_per_epoch, iteration_model_weights, device)

    lr_scheduler.step()

    train.validate(args, val_loader, model, criterion, device)

  iteration_model_weights.append(copy.deepcopy(model.state_dict()))
  torch.save(iteration_model_weights, args.weights_filename)

class MainArgs:
  epochs = 100
  start_epoch = 0
  batch_size = 128
  lr = 0.1
  momentum = 0.9
  weight_decay = 1e-4
  print_freq = 50
  workers = 1
  weights_filename = 'weights_frankle_seed_1_i=2048_seed_3'
  seed = 3
  stats_filename = f'stats_frankle_seed_1_i=2048_seed_3'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {DEVICE}')
args = MainArgs()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model = models.frankleResnet20().to(DEVICE)
# model.load_state_dict(torch.load('weights_frankle_seed_1_i=2048', map_location=DEVICE))
main(model, args, DEVICE)

