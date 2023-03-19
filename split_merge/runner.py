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
import os
from cli import shared_args
from environment import environment
from foundations import paths
from foundations.callbacks import is_logger_info_saved
from foundations.hparams import TrainingHparams
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
                self.desc.training_hparams, {'data_order_seed': seed})
            output_location = self.child_location(leg_i, seed)
            if models.registry.state_dicts_exist(self.child_location(leg_i, seed), self.desc.train_end_step):
                print(f'{indent}skipping training child with seed {seed}')
                continue

            pretrain_output_location = self.parent_location(leg_i)
            environment.exists_or_makedirs(output_location)
            model = models.registry.get(self.desc.model_hparams).to(environment.device())
            train.standard_train(
                model, output_location, 
                self.desc.dataset_hparams, training_hparams, 
                pretrain_output_location, self.desc.train_end_step, 
                save_dense=True)

    def _train_parent(self, leg_i):
        indent = " " * 1
        print(f'{indent}training parent')
        output_location = self.parent_location(leg_i)
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        if models.registry.state_dicts_exist(self.parent_location(leg_i), self.desc.train_end_step):
            print(f'{indent}parent already exists')
            return
        if leg_i == 0:
            train.standard_train(
                model, output_location, self.desc.dataset_hparams, 
                self.desc.training_hparams,
                save_dense=True)
        else:
            pretrain_output_location = self.avg_location(leg_i - 1)
            train.standard_train(
                model, output_location, self.desc.dataset_hparams, 
                self.desc.training_hparams, pretrain_output_location, 
                self.desc.train_end_step,
                save_dense=True)

    def _merge_children(self, leg_i):
        indent = " " * 2
        print(f'{indent}running average')
        # merge & save model state, optim state
        output_location = self.avg_location(leg_i)
        environment.exists_or_makedirs(output_location)
        if models.registry.model_exists(output_location, self.desc.train_end_step) and is_logger_info_saved(output_location, self.desc.train_end_step):
            print(f'{indent}average already exists')
            return

        standard_average(
            self.desc.dataset_hparams, self.desc.model_hparams, self.desc.training_hparams,
            self.avg_location(leg_i), self.leg_i_location(leg_i), 
            self.children_data_order_seeds, self.desc.train_end_step)

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