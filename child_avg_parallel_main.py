import argparse

from averaging.runner import AveragingRunner
from averaging.desc import AveragingDesc
from training.desc import TrainingDesc
from datasets import registry
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step

def main(step_i: int, parent_seed: int, child1_seed: int, child2_seed: int): 
    parent_training_hparams = TrainingHparams(data_order_seed=parent_seed)
    parent_dataset_hparams = DatasetHparams()
    parent_training_desc = TrainingDesc(dataset_hparams=parent_dataset_hparams, training_hparams=parent_training_hparams)

    parent_iterations_per_epoch = registry.get(parent_dataset_hparams).iterations_per_epoch
    parent_last_step = Step.from_str(parent_training_hparams.training_steps, parent_iterations_per_epoch)
    steps = Step.get_log_2_steps(parent_last_step, parent_iterations_per_epoch)

    child1_training_hparams = TrainingHparams(data_order_seed=child1_seed)
    child2_training_hparams = TrainingHparams(data_order_seed=child2_seed)
    child1_dataset_hparams = DatasetHparams()
    child2_dataset_hparams = DatasetHparams()

    if step_i >= len(steps):
        raise ValueError(f'given step_i {step_i} is out of range for {len(steps)} steps')
    print(f'running child avg at log step {step_i} for children seeds {child1_seed}, {child2_seed}')
    parent_step = steps[step_i]
    child1_training_desc = TrainingDesc(dataset_hparams=child1_dataset_hparams, training_hparams=child1_training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
    child2_training_desc = TrainingDesc(dataset_hparams=child2_dataset_hparams, training_hparams=child2_training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
    avg_runner = AveragingRunner(average_desc=AveragingDesc(train1=child1_training_desc, train2=child2_training_desc, dataset_hparams=parent_dataset_hparams))
    avg_runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_i', type=int)
    parser.add_argument('--parent_seed', type=int)
    parser.add_argument('--child1_seed', type=int)
    parser.add_argument('--child2_seed', type=int)
    args = parser.parse_args()
    main(args.step_i, args.parent_seed, args.child1_seed, args.child2_seed)