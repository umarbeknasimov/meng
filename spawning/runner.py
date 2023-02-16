import argparse
from dataclasses import dataclass
from cli import shared_args
from environment import environment
from foundations.callbacks import is_logger_info_saved
from foundations.runner import Runner
from foundations.hparams import TrainingHparams
from foundations import paths
from models.cifar_resnet import Model
from spawning.average import standard_average
from spawning.desc import SpawningDesc
from training import train
import models.registry

@dataclass
class SpawningRunner(Runner):
    desc: SpawningDesc
    children_data_order_seeds: list[int]
    experiment: str = 'main'

    @staticmethod
    def description():
        return 'spawn and train children from trained model'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        SpawningRunner._add_children_data_order_seeds_argument(parser)
        SpawningDesc.add_args(parser, shared_args.maybe_get_default_hparams())
    
    @staticmethod
    def _add_children_data_order_seeds_argument(parser: argparse.ArgumentParser):
        parser.add_argument('--children_data_order_seeds', type=int, nargs='+', required=True)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SpawningRunner':
        return SpawningRunner(
            SpawningDesc.create_from_args(args), 
            children_data_order_seeds=args.children_data_order_seeds, 
            experiment=args.experiment)
    
    def _train(self):
        location = self._train_location()
        environment.exists_or_makedirs(location)
        
        if models.registry.state_dicts_exist(location, self.desc.train_end_step): 
            print('train model already exists')
            return

        print('not all spawn steps saved so running train on parent')
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        if self.desc.pretrain_dataset_hparams and self.desc.pretrain_training_hparams:
            pretrain_output_location = self._pretrain_location()
            # load model weights from pretrained model
            train.standard_train(
                model, location, self.desc.dataset_hparams, 
                self.desc.training_hparams, pretrain_output_location, 
                self.desc.pretrain_end_step,
                pretrain_load_only_model_weights=True)
        else:
            train.standard_train(
                model, location, self.desc.dataset_hparams, 
                self.desc.training_hparams)
    
    def _pretrain(self):
        output_location = self._pretrain_location()
        if models.registry.state_dicts_exist(output_location, self.desc.pretrain_end_step): 
            print('pretrain model already exists')
            return
        print('pretrain model doesn\'t exist so running pretrain')
    
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        train.standard_train(model, output_location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams)
    
    def _spawn_and_train(self, spawn_step, data_order_seed):
        training_hparams = TrainingHparams.create_from_instance_and_dict(
            self.desc.training_hparams, {'data_order_seed': data_order_seed})
        output_location = self._spawn_step_child_location(spawn_step, data_order_seed)
        print(f'child at spawn step {spawn_step.ep_it_str} with seed {data_order_seed}')
        if models.registry.state_dicts_exist(output_location, self.desc.train_end_step):
            print(f'child already exists')
            return
        print(f'child doesn\'t exist so running train')
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        train.standard_train(model, output_location, self.desc.dataset_hparams, training_hparams, self._train_location(), spawn_step)
    
    def _average(self, spawn_step, seeds):
        print(f'averaging children for seeds {seeds} at spawn step {spawn_step.ep_it_str}')

        output_location = self._spawn_step_average_location(spawn_step, seeds)
        spawn_step_location = self._spawn_step_location(spawn_step)

        for child_step in self.desc.children_saved_steps:
            print(f'child step {child_step.ep_it_str}')
            if models.registry.model_exists(output_location, child_step) and is_logger_info_saved(output_location, child_step):
                print('not running average')
                continue
            print('running average')
            standard_average(self.desc.dataset_hparams, self.desc.model_hparams, output_location, spawn_step_location, seeds, child_step)
                
    def run(self, spawn_step_index: int = None):
        print(f'running {self.description()}')

        self.desc.save_hparam(self.desc.run_path(part='main', experiment=self.experiment))
        if self.desc.pretrain_dataset_hparams and self.desc.pretrain_training_hparams: self._pretrain()
        
        self._train()

        print(f'spawning children with seeds {self.children_data_order_seeds}')
        indices = []
        # for running parallel slurm job tasks
        if spawn_step_index:
            print(f'running for only spawn step index {spawn_step_index}')
            indices = [spawn_step_index, spawn_step_index + 1]
        else:
            print(f'running for all spawn steps')
            indices = [0, len(self.desc.spawn_steps)]
        for spawn_step_i in range(*indices):
            spawn_step = self.desc.spawn_steps[spawn_step_i]
            for data_order_seed in self.children_data_order_seeds:
                self._spawn_and_train(spawn_step, data_order_seed)
            if len(self.children_data_order_seeds) > 1:
                self._average(spawn_step, self.children_data_order_seeds)

    def _train_location(self):
        return self.desc.run_path(part='parent', experiment=self.experiment)
    
    def _pretrain_location(self):
        return self.desc.run_path(part='pretrain', experiment=self.experiment)
    
    def _spawn_step_location(self, spawn_step):
        return paths.spawn_step(self.desc.run_path(part='children', experiment=self.experiment), spawn_step)
    
    def _spawn_step_average_location(self, spawn_step, seeds):
        return paths.average(self._spawn_step_location(spawn_step), seeds)
    
    def _spawn_step_child_location(self, spawn_step, data_order_seed):
        return paths.seed(self._spawn_step_location(spawn_step), data_order_seed)


                

        
