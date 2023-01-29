"""
generates and saves children models based on some model's training run

let's say a model has iterates: W_0, W_1, ..., W_f, where W_i is iteration i
we usually keep track of W_0, W_1, W_2, W_4, W_8, ... W_{2**i} for space efficiency
for each W, generate 2 copies (children) with seed 1 and seed 2
train each child with learning rate 0.1 for 4096 iterations (10 epochs)
save the weights of each child (every 2**i iterations)

the goal of this:
 to analyze the mode connectivity of the children through various i
 to find critical i
"""
import os

from utils import load
import torch
import models
from environment import environment
from foundations.hparams import TrainingHParams
from foundations.step import Step
from training.train import train
from training.callbacks import standard_callbacks
import dataset
import math
import ssl


def main():
    PARENT_SEED = 0
    CHILD_SEEDS = [1]

    ssl._create_default_https_context = ssl._create_unverified_context

    train_loader, _ = dataset.get_train_test_loaders()
    iterations_per_epoch = len(train_loader)
    EXPONENTIAL_STEPS = [Step.zero(iterations_per_epoch)] + [Step.from_iteration(2**i, iterations_per_epoch) for i in range(int(math.log2(100*iterations_per_epoch)))]

    parent_file = os.path.join(environment.get_user_dir(), 'parent', 's_{}'.format(PARENT_SEED))
    if not os.path.exists(parent_file):
        raise ValueError('parent directory doesn\'t exist')
    children_dir = os.path.join(parent_file, 'children')
    if not os.path.exists(children_dir):
        os.makedirs(children_dir)
        print('making children dir {}'.format(children_dir))


    for seed_i in CHILD_SEEDS:
        for step in EXPONENTIAL_STEPS:
            parent_state_dict = torch.load(os.path.join(parent_file, 'ep{}_it{}.pth'.format(step.ep, step.it)))
            children_dir = os.path.join(children_dir, 'ep{}_it{}'.format(step.ep, step.it))
            child_output_location = os.path.join(children_dir, 's_{}'.format(seed_i))
            if not os.path.exists(child_output_location):
                print('making children output location {}'.format(child_output_location))
                os.makedirs(child_output_location)

            torch.manual_seed(seed_i)
            torch.cuda.manual_seed(seed_i)
            train_loader, test_loader = dataset.get_train_test_loaders()
            args = TrainingHParams(seed=seed_i)
            
            model = models.frankleResnet20().to(environment.device())
            model.load_state_dict(parent_state_dict['model'])
            print('training child spawned from iteration {} with seed {}'.format(step.iteration, seed_i))
            train(model, args, standard_callbacks(args, train_loader, test_loader), child_output_location, train_loader, parent_state_dict['optimizer'], parent_state_dict['scheduler'])

if __name__ == '__main__':
    main()