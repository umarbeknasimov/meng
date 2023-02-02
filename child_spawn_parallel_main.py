import argparse
import sys

from training.desc import TrainingDesc
from training.runner import TrainingRunner
from datasets import registry
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step

def main(step_i: int, parent_seed: int, child_seed: int): 
    parent_training_hparams = TrainingHparams(data_order_seed=parent_seed)
    parent_dataset_hparams = DatasetHparams()
    parent_training_desc = TrainingDesc(dataset_hparams=parent_dataset_hparams, training_hparams=parent_training_hparams)

    parent_iterations_per_epoch = registry.get(parent_dataset_hparams).iterations_per_epoch
    parent_last_step = Step.from_str(parent_training_hparams.training_steps, parent_iterations_per_epoch)
    steps = Step.get_log_2_steps(parent_last_step, parent_iterations_per_epoch)

    training_hparams = TrainingHparams(data_order_seed=child_seed)
    dataset_hparams = DatasetHparams()

    if step_i >= len(steps):
        raise ValueError(f'given step_i {step_i} is out of range for {len(steps)} steps')
    print(f'running child at log step {step_i} for child seed {child_seed}')
    parent_step = steps[step_i]
    training_desc = TrainingDesc(dataset_hparams=dataset_hparams, training_hparams=training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
    runner = TrainingRunner(training_desc=training_desc)
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_i', type=int)
    parser.add_argument('--parent_seed', type=int)
    parser.add_argument('--child_seed', type=int)
    args = parser.parse_args()
    main(args.step_i, args.parent_seed, args.child_seed)
