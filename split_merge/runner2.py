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
class SplitMergeRunner2:
    desc: SplitMergeDesc
    num_legs: int = 10
    experiment: str = 'main'

    @staticmethod
    def description():
        return 'train a model by splitting and merging'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        SplitMergeRunner2._add_extra_params(parser)
        SplitMergeDesc.add_args(parser, shared_args.maybe_get_default_hparams())
    
    @staticmethod
    def _add_extra_params(parser: argparse.ArgumentParser):
        parser.add_argument('--num_legs', type=int, required=True)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SplitMergeRunner2':
        return SplitMergeRunner2(
            SplitMergeDesc.create_from_args(args), 
            num_legs=args.num_legs,
            experiment=args.experiment)
    
    def _train_child(self, leg_i):
        # train children for s steps
        indent = " " * 2
        print(f'{indent}training child with seed')

        output_location = self.child_location(leg_i)
        if models.registry.state_dicts_exist(self.child_location(leg_i), self.child_train_end_step(leg_i)):
            print(f'{indent}skipping training child')
            return

        pretrain_output_location = self.parent_location(leg_i)
        environment.exists_or_makedirs(output_location)
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        if self.desc.strategy == 'restart_optimizer':
            train.standard_train(
                model, output_location, 
                self.child_dataset_hparams(leg_i), self.desc.training_hparams, 
                pretrain_output_location, self.parent_train_end_step(leg_i), 
                pretrain_load_only_model_weights=True, evaluate_every_10=True)
        else:
            train.standard_train(
                model, output_location, 
                self.child_dataset_hparams(leg_i), self.desc.training_hparams, 
                pretrain_output_location, self.parent_train_end_step(leg_i), evaluate_every_10=True)

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
                training_hparams, evaluate_every_10=True)
        else:
            pretrain_output_location = self.avg_location(leg_i - 1)
            if self.desc.strategy == 'restart_optimizer':
                print('restarting optimizer')
                train.standard_train(
                    model, output_location, self.parent_dataset_hparams(leg_i), 
                    training_hparams, pretrain_output_location, 
                    self.child_train_end_step(leg_i - 1),
                    pretrain_load_only_model_weights=True, evaluate_every_10=True)
            else:
                train.standard_train(
                    model, output_location, self.parent_dataset_hparams(leg_i), 
                    training_hparams, pretrain_output_location, 
                    self.child_train_end_step(leg_i - 1), evaluate_every_10=True)

    def _merge(self, leg_i):
        indent = " " * 2
        print(f'{indent}running average')

        output_location = self.avg_location(leg_i)
        end_step = self.child_train_end_step(leg_i)
        child_location = self.child_location(leg_i)
        
        all_steps = Step.get_log_2_steps_equally_spaced_by_0point5(end_step)

        start_step = Step.zero(datasets.registry.get_iterations_per_epoch(self.desc.dataset_hparams))
        start_weights = start_weights = models.registry.get_model_state_dict(
            child_location,
            start_step)
        end_weights = models.registry.get_model_state_dict(
            child_location,
            end_step)
        
        other_weights = [
            models.registry.get_model_state_dict(
            child_location,
                all_steps[-3]),
            models.registry.get_model_state_dict(
            child_location,
                all_steps[-5])
        ]

        start_optim = models.registry.get_optim_state_dict(
            child_location,
            end_step)['optimizer']
        
        # keep optim
        end_optim = models.registry.get_optim_state_dict(
            child_location,
            end_step)['optimizer']

        other_optim = [
            models.registry.get_optim_state_dict(
            child_location,
                all_steps[-3])['optimizer'],
            models.registry.get_optim_state_dict(
            child_location,
                all_steps[-5])['optimizer']
        ]

        weights = [start_weights, end_weights] + other_weights
        optims = [start_optim, end_optim] + other_optim
        standard_average(self.desc.dataset_hparams,
            self.desc.model_hparams,
            self.desc.training_hparams,
            output_location,
            end_step,
            weights,
            optims)
    

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
            self._train_child(leg_i)
            self._merge(leg_i)
    
    def leg_location(self):
        return self.desc.run_path('legs', self.experiment)
    
    def leg_i_location(self, leg_i):
        return os.path.join(self.leg_location(), str(leg_i))

    def parent_location(self, leg_i):
        return os.path.join(self.leg_i_location(leg_i), 'parent')

    def child_location(self, leg_i):
        return os.path.join(self.leg_i_location(leg_i), 'child')
    
    def avg_location(self, leg_i):
        return paths.average_no_seeds(self.leg_i_location(leg_i))