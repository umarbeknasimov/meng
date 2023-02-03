import argparse

from averaging.desc import AveragingDesc
import json
from environment import environment
import os
from training.desc import TrainingDesc
from datasets import registry
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step
from foundations import paths

def main(parent_seed: int, child1_seed: int, child2_seed: int): 
    result = {}
    parent_training_hparams = TrainingHparams(data_order_seed=parent_seed)
    parent_dataset_hparams = DatasetHparams()
    parent_training_desc = TrainingDesc(dataset_hparams=parent_dataset_hparams, training_hparams=parent_training_hparams)
    save_logger_from_desc(parent_training_desc, result)

    parent_iterations_per_epoch = registry.get(parent_dataset_hparams).iterations_per_epoch
    parent_last_step = Step.from_str(parent_training_hparams.training_steps, parent_iterations_per_epoch)
    steps = Step.get_log_2_steps(parent_last_step, parent_iterations_per_epoch)

    child1_training_hparams = TrainingHparams(data_order_seed=child1_seed)
    child2_training_hparams = TrainingHparams(data_order_seed=child2_seed)
    child1_dataset_hparams = DatasetHparams()
    child2_dataset_hparams = DatasetHparams()

    for parent_step in steps:
        print(f'getting info for parent step {parent_step}')
        child1_training_desc = TrainingDesc(dataset_hparams=child1_dataset_hparams, training_hparams=child1_training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
        child2_training_desc = TrainingDesc(dataset_hparams=child2_dataset_hparams, training_hparams=child2_training_hparams, pretrain_training_desc=parent_training_desc, pretrain_step=f'{parent_step.ep}ep{parent_step.it}it')
        save_logger_from_desc(child1_training_desc, result)
        save_logger_from_desc(child2_training_desc, result)
        avg_desc = AveragingDesc(train1=child1_training_desc, train2=child2_training_desc, dataset_hparams=parent_dataset_hparams)
        save_logger_from_desc(avg_desc, result)
    
    with open(os.path.join(environment.get_user_dir(), 'report.json'), 'w') as f:
        json_string = json.dumps(result)
        json.dump(json_string, f)

def save_logger_from_desc(desc, result):
    if not os.path.exists(desc.run_path()):
        print(f'path to desc \n {desc} does not exist')
    with open(paths.logger(desc.run_path()), 'r') as f:
        result[desc.hashname] = f.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_seed', type=int)
    parser.add_argument('--child1_seed', type=int)
    parser.add_argument('--child2_seed', type=int)
    args = parser.parse_args()
    main(args.parent_seed, args.child1_seed, args.child2_seed)