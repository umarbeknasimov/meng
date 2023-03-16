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
from foundations.hparams import TrainingHparams
import models.registry
import datasets.registry
from split_merge.desc import SplitMergeDesc
from training import train
from utils import interpolate, state_dict

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
    
    def train_children(self, leg_i):
        # train children for s steps
        for seed in self.children_data_order_seeds:
            training_hparams = TrainingHparams.create_from_instance_and_dict(
                self.desc.training_hparams, {'data_order_seed': seed})
            output_location = self.child_location(leg_i, seed)
            pretrain_output_location = self.parent_location(leg_i)
            environment.exists_or_makedirs(output_location)
            print(f'  training child with seed {seed}')
            model = models.registry.get(self.desc.model_hparams).to(environment.device())
            train.standard_train(
                model, output_location, 
                self.desc.dataset_hparams, training_hparams, 
                pretrain_output_location, self.desc.train_end_step, 
                save_dense=True)
    
    def leg_location(self, leg_i):
        return self.desc.run_path(
            os.path.join('legs', str(leg_i)), self.experiment)

    def parent_location(self, leg_i):
        return os.path.join(self.leg_location(leg_i), 'parent')

    def child_location(self, leg_i, data_order_seed):
        return paths.seed(self.leg_location(leg_i), data_order_seed)
    
    def avg_location(self, leg_i):
        return paths.average_no_seeds(self.leg_location(leg_i))

    def train_parent(self, leg_i):
        print(f' training parent')
        output_location = self.parent_location(leg_i)
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
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

    def merge_children(self, leg_i, train_loader):
        # merge & save model state, optim state
        environment.exists_or_makedirs(self.avg_location(leg_i))

        # merge weights
        weights = []
        for seed in self.children_data_order_seeds:
            weights.append(
                environment.load(
                    paths.model(self.child_location(leg_i, seed), self.desc.train_end_step))
            )
        
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        averaged_weights = interpolate.average_state_dicts(weights)
        averaged_weights_wo_batch_stats = state_dict.get_state_dict_wo_batch_stats(
            model, averaged_weights)
        model.load_state_dict(averaged_weights_wo_batch_stats)
        interpolate.forward_pass(model, train_loader)
        environment.save(model.state_dict(), paths.model(self.avg_location(leg_i), self.desc.train_end_step))

        # merge optim
        optimizers = []
        for seed in self.children_data_order_seeds:
            optimizers.append(
                environment.load(
                    paths.optim(self.child_location(leg_i, seed), self.desc.train_end_step)
                )['optimizer']
            )
        averaged_optimizers = interpolate.average_optimizer_state_dicts(optimizers)
        environment.save({
            'optimizer': averaged_optimizers,
            'scheduler': None
        }, paths.optim(self.avg_location(leg_i), self.desc.train_end_step))


    def run(self):
        # train
        main_path = self.desc.run_path(part='main', experiment=self.experiment)
        environment.exists_or_makedirs(main_path)
        self.desc.save_hparam(main_path)

        train_loader = datasets.registry.get(self.desc.dataset_hparams)

        for leg_i in range(self.num_legs):
            print(f'running leg {leg_i}')
            self.train_parent(leg_i)
            self.train_children(leg_i)
            self.merge_children(leg_i, train_loader)
