import sys

from averaging.runner import AveragingRunner
from averaging.desc import AveragingDesc
from training.desc import TrainingDesc
from datasets import registry
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step

def main(step_i: int): 
    PARENT_SEED = 1
    CHILD1_SEED = 2
    CHILD2_SEED = 3

    parent_training_hparams = TrainingHparams(data_order_seed=PARENT_SEED)
    parent_dataset_hparams = DatasetHparams()
    parent_training_desc = TrainingDesc(dataset_hparams=parent_dataset_hparams, training_hparams=parent_training_hparams)

    parent_iterations_per_epoch = registry.get(parent_dataset_hparams).iterations_per_epoch
    parent_last_step = Step.from_str(parent_training_hparams.training_steps, parent_iterations_per_epoch)
    steps = Step.get_log_2_steps(parent_last_step, parent_iterations_per_epoch)

    child1_training_hparams = TrainingHparams(data_order_seed=CHILD1_SEED)
    child2_training_hparams = TrainingHparams(data_order_seed=CHILD2_SEED)
    child1_dataset_hparams = DatasetHparams()
    child2_dataset_hparams = DatasetHparams()

    if step_i >= len(steps):
        raise ValueError(f'given step_i {step_i} is out of range for {len(steps)} steps')
    print(f'running child avg at log step {step_i} for children seeds {CHILD1_SEED}, {CHILD2_SEED}')
    parent_step = steps[step_i]
    child1_training_desc = TrainingDesc(dataset_hparams=child1_dataset_hparams, training_hparams=child1_training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
    child2_training_desc = TrainingDesc(dataset_hparams=child2_dataset_hparams, training_hparams=child2_training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
    avg_runner = AveragingRunner(AveragingDesc(child1_training_desc, child2_training_desc, parent_dataset_hparams))
    avg_runner.run()

if __name__ == "__main__":
    step_i = int(sys.argv[1])
    main(step_i)