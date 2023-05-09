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
    k: str = '1024it'

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
        parser.add_argument('--k', type=str, required=True)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SplitMergeRunner':
        return SplitMergeRunner(
            SplitMergeDesc.create_from_args(args), 
            children_data_order_seeds=args.children_data_order_seeds, 
            num_legs=args.num_legs,
            experiment=args.experiment,
            k=args.k)
    
    def _train_children(self, leg_i):
        # train children for s steps
        indent = " " * 2
        for seed in self.children_data_order_seeds:
            print(f'{indent}training child with seed {seed}')
            training_hparams = TrainingHparams.create_from_instance_and_dict(
                self.desc.training_hparams, {'data_order_seed': seed})

            output_location = self.child_location(leg_i, seed)
            if models.registry.state_dicts_exist(self.child_location(leg_i, seed), self.child_train_end_step):
                print(f'{indent}skipping training child with seed {seed}')
                continue

            if leg_i == 0:
                pretrain_output_location = self.parent_location
                end_step = self.parent_train_end_step
            else:
                pretrain_output_location = self.avg_location(leg_i - 1)
                end_step = self.child_train_end_step
            environment.exists_or_makedirs(output_location)
            model = models.registry.get(self.desc.model_hparams).to(environment.device())
            train.standard_train(
                model, output_location, 
                self.desc.dataset_hparams, training_hparams, 
                pretrain_output_location, end_step)

    def _train_parent(self):
        indent = " " * 1
        print(f'{indent}training parent')
        output_location = self.parent_location
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        if models.registry.state_dicts_exist(self.parent_location, self.parent_train_end_step):
            print(f'{indent}parent already exists')
            return

        training_hparams = TrainingHparams.create_from_instance_and_dict(
            self.desc.training_hparams, {'training_steps': self.k})
            
        train.standard_train(
            model, output_location, self.desc.dataset_hparams, 
            training_hparams)

    def _merge_children(self, leg_i):
        indent = " " * 2
        print(f'{indent}running average')

        output_location = self.avg_location(leg_i)
        step = self.child_train_end_step

        if models.registry.model_exists(output_location, self.child_train_end_step) and is_logger_info_saved(output_location, self.child_train_end_step):
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
    
    @property
    def parent_train_end_step(self):
        return Step.from_str(
            self.k, 
            datasets.registry.get(self.desc.dataset_hparams).iterations_per_epoch)

    @property
    def child_train_end_step(self):
        return Step.from_str(
            self.desc.training_hparams.training_steps, 
            datasets.registry.get(self.desc.dataset_hparams).iterations_per_epoch)

    def run(self):
        # train
        main_path = self.desc.run_path(part='main', experiment=self.experiment)
        environment.exists_or_makedirs(main_path)
        self.desc.save_hparam(main_path)

        self._train_parent()

        for leg_i in range(self.num_legs):
            print(f'running leg {leg_i}')
            self._train_children(leg_i)
            self._merge_children(leg_i)
    
    def leg_location(self):
        return self.desc.run_path('legs', self.experiment)
    
    def leg_i_location(self, leg_i):
        return os.path.join(self.leg_location(), str(leg_i))

    @property
    def parent_location(self):
        return os.path.join(self.leg_i_location(0), 'parent')

    def child_location(self, leg_i, data_order_seed):
        return paths.seed(self.leg_i_location(leg_i), data_order_seed)
    
    def avg_location(self, leg_i):
        return paths.average_no_seeds(self.leg_i_location(leg_i))
