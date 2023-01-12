import torch
import torch.nn as nn
import copy

import dataset
import train

def main(model, args, device):
  iteration_model_weights = []

  train_loader, val_loader = dataset.get_train_val_loaders()
  
  steps_per_epoch = len(train_loader)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=[33, 66], last_epoch=-1)
  for epoch in range(args.epochs):

    # train for one epoch
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train.train_epoch(args, train_loader, model, criterion, 
                                  optimizer, epoch, steps_per_epoch, iteration_model_weights, device)

    lr_scheduler.step()

    train.validate(args, val_loader, model, criterion, device)

  iteration_model_weights.append(copy.deepcopy(model.state_dict()))
  torch.save(iteration_model_weights, args.weights_filename)

