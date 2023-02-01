from training.desc import TrainingDesc
from training.runner import TrainingRunner

import torch
import math
import sys

import dataset
from foundations.hparams import ModelHparams, TrainingHparams
from foundations.step import Step

def main(step: Step): 
    PARENT_SEED = 2
    CHILD_SEED = 1

    training_hparams = TrainingHparams(seed=CHILD_SEED)

    torch.manual_seed(CHILD_SEED)
    torch.cuda.manual_seed(CHILD_SEED)
    model_hparams = ModelHparams(init_step=step, init_step_seed=PARENT_SEED)
    training_desc = TrainingDesc(model_hparams=model_hparams, training_hparams=training_hparams)
    runner = TrainingRunner(training_desc=training_desc)
    print('training child spawned from iteration {} with seed {}'.format(step.iteration, CHILD_SEED))
    runner.run()

if __name__ == "__main__":
    index = int(sys.argv[1])
    train_loader, _ = dataset.get_train_test_loaders()
    iterations_per_epoch = len(train_loader)
    step_i = ([Step.zero(iterations_per_epoch)] + [Step.from_iteration(2**i, iterations_per_epoch) for i in range(int(math.log2(100*iterations_per_epoch)))])[index]
    main(step_i)