from training.desc import TrainingDesc
from training.runner import TrainingRunner
from datasets import registry

import math
import sys

from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step

def main(step_i: int): 
    PARENT_SEED = 1
    CHILD_SEED = 2

    parent_training_hparams = TrainingHparams(data_order_seed=PARENT_SEED)
    parent_dataset_hparams = DatasetHparams()
    parent_training_desc = TrainingDesc(dataset_hparams=parent_dataset_hparams, training_hparams=parent_training_hparams)

    parent_iterations_per_epoch = registry.get(parent_dataset_hparams).iterations_per_epoch
    parent_last_step = Step.from_str(parent_training_hparams.training_steps, parent_iterations_per_epoch)
    steps = ([Step.zero(parent_iterations_per_epoch)] + [Step.from_iteration(2**i, parent_iterations_per_epoch) for i in range(int(math.log2(parent_last_step.iteration)))])

    training_hparams = TrainingHparams(data_order_seed=CHILD_SEED)
    dataset_hparams = DatasetHparams()

    if step_i >= len(steps):
        raise ValueError(f'given step_i {step_i} is out of range for {len(steps)} steps')
    print(f'running child at log step {step_i} for child seed {CHILD_SEED}')
    parent_step = steps[step_i]
    training_desc = TrainingDesc(dataset_hparams=dataset_hparams, training_hparams=training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
    runner = TrainingRunner(training_desc=training_desc)
    runner.run()


if __name__ == "__main__":
    step_i = int(sys.argv[1])
    main(step_i)
