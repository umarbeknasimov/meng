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
from training.desc import TrainingDesc
from training.runner import TrainingRunner

import torch
import math
import ssl

import datasets.dataset as dataset
from foundations.hparams import ModelHparams, TrainingHparams
from foundations.step import Step

def main() :
    ssl._create_default_https_context = ssl._create_unverified_context
    PARENT_SEED = 2
    CHILD_SEED = 1

    training_hparams = TrainingHparams(seed=CHILD_SEED)
    train_loader, _ = dataset.get_train_test_loaders()
    iterations_per_epoch = len(train_loader)
    EXPONENTIAL_STEPS = [Step.zero(iterations_per_epoch)] + [Step.from_iteration(2**i, iterations_per_epoch) for i in range(int(math.log2(100*iterations_per_epoch)))]

    for step in EXPONENTIAL_STEPS:
        torch.manual_seed(CHILD_SEED)
        torch.cuda.manual_seed(CHILD_SEED)
        model_hparams = ModelHparams(init_step=step, init_step_seed=PARENT_SEED)
        training_desc = TrainingDesc(model_hparams=model_hparams, training_hparams=training_hparams)
        runner = TrainingRunner(training_desc=training_desc)
        print('training child spawned from iteration {} with seed {}'.format(step.iteration, CHILD_SEED))
        runner.run()


if __name__ == '__main__':
    main()