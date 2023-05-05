# children_seeds: list[int]
# s: number of steps to train per parent/child leg
    # can make this part of the training_hparams
# c: number of legs

# dir:
    # legs
        # 0
            # parent
                # model0ep0it
                # optim0ep0it
            # seed1
                # model0...
            # seed2
            # avg
                # model0...

        # 1
        # 2

import argparse
from dataclasses import dataclass
import datasets.registry
import os
from cli import shared_args
from environment import environment
from foundations import paths
from foundations.callbacks import is_logger_info_saved
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step
import models.registry
from spawning.average import standard_average
from split_merge.desc import SplitMergeDesc
from training import train

@dataclass
class SplitMergeRunner:
    desc: SplitMergeDesc
    children_data_order_seeds: list
    num_legs: int = 10
    experiment: str = 'main'

    @staticmethod
    def description():
        return 'train a model by splitting and merging'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        SplitMergeRunner._add_extra_params(parser)
        SplitMergeDesc.add_args(parser, shared_args.maybe_get_default_hparams())
    
    @staticmethod
    def _add_extra_params(parser: argparse.ArgumentParser):
        parser.add_argument('--children_data_order_seeds', type=int, nargs='+', required=True)
        parser.add_argument('--num_legs', type=int, required=True)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SplitMergeRunner':
        return SplitMergeRunner(
            SplitMergeDesc.create_from_args(args), 
            children_data_order_seeds=args.children_data_order_seeds, 
            num_legs=args.num_legs,
            experiment=args.experiment)
    
    def _train_children(self, leg_i):
        # train children for s steps
        indent = " " * 2
        for seed in self.children_data_order_seeds:
            print(f'{indent}training child with seed {seed}')
            training_hparams = TrainingHparams.create_from_instance_and_dict(
                self.training_hparams(leg_i), {'data_order_seed': seed})

            output_location = self.child_location(leg_i, seed)
            if models.registry.state_dicts_exist(self.child_location(leg_i, seed), self.child_train_end_step(leg_i)):
                print(f'{indent}skipping training child with seed {seed}')
                continue

            pretrain_output_location = self.parent_location(leg_i)
            environment.exists_or_makedirs(output_location)
            model = models.registry.get(self.desc.model_hparams).to(environment.device())
            if self.desc.strategy == 'restart_optimizer':
                train.standard_train(
                    model, output_location, 
                    self.child_dataset_hparams(leg_i), training_hparams, 
                    pretrain_output_location, self.parent_train_end_step(leg_i), 
                    pretrain_load_only_model_weights=True)
            else:
                train.standard_train(
                    model, output_location, 
                    self.child_dataset_hparams(leg_i), training_hparams, 
                    pretrain_output_location, self.parent_train_end_step(leg_i))

    def _train_parent(self, leg_i):
        indent = " " * 1
        print(f'{indent}training parent')
        output_location = self.parent_location(leg_i)
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        if models.registry.state_dicts_exist(self.parent_location(leg_i), self.parent_train_end_step(leg_i)):
            print(f'{indent}parent already exists')
            return
        
        training_hparams = self.training_hparams(leg_i)
        
        if leg_i == 0:
            train.standard_train(
                model, output_location, self.parent_dataset_hparams(0), 
                training_hparams)
        else:
            pretrain_output_location = self.avg_location(leg_i - 1)
            if self.desc.strategy == 'restart_optimizer':
                print('restarting optimizer')
                train.standard_train(
                    model, output_location, self.parent_dataset_hparams(leg_i), 
                    training_hparams, pretrain_output_location, 
                    self.child_train_end_step(leg_i - 1),
                    pretrain_load_only_model_weights=True)
            else:
                train.standard_train(
                    model, output_location, self.parent_dataset_hparams(leg_i), 
                    training_hparams, pretrain_output_location, 
                    self.child_train_end_step(leg_i - 1))

    def _merge_children(self, leg_i):
        indent = " " * 2
        print(f'{indent}running average')
        # merge & save model state, optim state
        # output_location = self.avg_location(leg_i)
        # environment.exists_or_makedirs(output_location)
        # if models.registry.model_exists(output_location, self.child_train_end_step(leg_i)) and is_logger_info_saved(output_location, self.child_train_end_step(leg_i)):
        #     print(f'{indent}average already exists')
        #     return

        # standard_average(
        #     self.desc.dataset_hparams, self.desc.model_hparams, self.desc.training_hparams,
        #     self.avg_location(leg_i), self.leg_i_location(leg_i), 
        #     self.children_data_order_seeds, self.child_train_end_step(leg_i))

        output_location = self.avg_location(leg_i)
        step = self.child_train_end_step(leg_i)

        if models.registry.model_exists(output_location, self.child_train_end_step(leg_i)) and is_logger_info_saved(output_location, self.child_train_end_step(leg_i)):
            print(f'{indent}average already exists')
            return
        children_weights = []
        children_optimizer_weights = []
        for seed_i in self.children_data_order_seeds:
            child_weights = models.registry.get_model_state_dict(
                paths.seed(self.leg_i_location(leg_i), seed_i),
                step)
            child_optimizer_weights = models.registry.get_optim_state_dict(
                paths.seed(self.leg_i_location(leg_i), seed_i),
                step)['optimizer']
            children_weights.append(child_weights)
            children_optimizer_weights.append(child_optimizer_weights)
        standard_average(self.desc.dataset_hparams,
            self.desc.model_hparams,
            self.desc.training_hparams,
            output_location,
            step,
            children_weights,
            children_optimizer_weights)
    
    def training_hparams(self, leg_i):
        if self.desc.strategy == 'decrease_lr':
            training_hparams = TrainingHparams.create_from_instance_and_dict(
                self.desc.training_hparams, 
                {
                    'lr': round(self.desc.training_hparams.lr / (4 ** leg_i), 8)
                })
        else:
            training_hparams = self.desc.training_hparams
        return training_hparams
    
    def child_dataset_hparams(self, leg_i):
        if self.desc.strategy == 'subsample':
            dataset_hparams = DatasetHparams.create_from_instance_and_dict(
                self.desc.dataset_hparams, 
                {
                    'subsample_fraction': 1/len(self.children_data_order_seeds)
                })
        elif self.desc.strategy == 'increase_batch_size':
            dataset_hparams = DatasetHparams.create_from_instance_and_dict(
                self.desc.dataset_hparams, 
                {
                    'batch_size': int(self.desc.dataset_hparams.batch_size * (len(self.children_data_order_seeds)**leg_i))
                })
        elif self.desc.strategy == 'decrease_child_batch_size':
            dataset_hparams = DatasetHparams.create_from_instance_and_dict(
                self.desc.dataset_hparams, 
                {
                    'batch_size': int(self.desc.dataset_hparams.batch_size / len(self.children_data_order_seeds))
                })
        else:
            dataset_hparams = self.desc.dataset_hparams
        return dataset_hparams
    
    def parent_dataset_hparams(self, leg_i):
        if self.desc.strategy == 'increase_batch_size':
            dataset_hparams = DatasetHparams.create_from_instance_and_dict(
                self.desc.dataset_hparams, 
                {
                    'batch_size': int(self.desc.dataset_hparams.batch_size * (len(self.children_data_order_seeds)**leg_i))
                })
        else:
            dataset_hparams = self.desc.dataset_hparams
        return dataset_hparams
    
    def end_step(self, leg_i, of_parent=True):
        return Step.from_str(
            self.desc.training_hparams.training_steps, 
            datasets.registry.get(
                self.parent_dataset_hparams(leg_i) if of_parent else self.child_dataset_hparams(leg_i)).iterations_per_epoch)
    
    def parent_train_end_step(self, leg_i):
        return self.end_step(leg_i)

    def child_train_end_step(self, leg_i):
        return self.end_step(leg_i, False)

    def run(self):
        # train
        main_path = self.desc.run_path(part='main', experiment=self.experiment)
        environment.exists_or_makedirs(main_path)
        self.desc.save_hparam(main_path)

        for leg_i in range(self.num_legs):
            print(f'running leg {leg_i}')
            self._train_parent(leg_i)
            self._train_children(leg_i)
            self._merge_children(leg_i)
    
    def leg_location(self):
        return self.desc.run_path('legs', self.experiment)
    
    def leg_i_location(self, leg_i):
        return os.path.join(self.leg_location(), str(leg_i))

    def parent_location(self, leg_i):
        return os.path.join(self.leg_i_location(leg_i), 'parent')

    def child_location(self, leg_i, data_order_seed):
        return paths.seed(self.leg_i_location(leg_i), data_order_seed)
    
    def avg_location(self, leg_i):
        return paths.average_no_seeds(self.leg_i_location(leg_i))
