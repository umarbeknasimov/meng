import argparse
from dataclasses import dataclass
from cli import shared_args
import datasets.registry
from environment import environment
from foundations.callbacks import is_logger_info_saved
from foundations.runner import Runner
from foundations.hparams import TrainingHparams
from foundations import paths
from foundations.step import Step
from spawning.average import standard_average
from spawning.desc import SpawningDesc
from training import train
import models.registry

@dataclass
class SpawningRunner(Runner):
    desc: SpawningDesc
    children_data_order_seeds: list
    experiment: str = 'main'
    save_dense: bool = False

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
            experiment=args.experiment,
            save_dense=args.save_dense)
    
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
                pretrain_load_only_model_weights=True,
                save_dense=self.save_dense)
        else:
            train.standard_train(
                model, location, self.desc.dataset_hparams, 
                self.desc.training_hparams,
                save_dense=self.save_dense)
    
    def _pretrain(self):
        output_location = self._pretrain_location()
        environment.exists_or_makedirs(output_location)
        if models.registry.state_dicts_exist(output_location, self.desc.pretrain_end_step): 
            print('pretrain model already exists')
            return
        print('pretrain model doesn\'t exist so running pretrain')
    
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        train.standard_train(model, output_location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams, save_dense=self.save_dense)
    
    def _spawn_and_train(self, spawn_step, data_order_seed):
        training_hparams = TrainingHparams.create_from_instance_and_dict(
            self.desc.training_hparams, {'data_order_seed': data_order_seed})
        output_location = self._spawn_step_child_location(spawn_step, data_order_seed)
        environment.exists_or_makedirs(output_location)
        print(f'child at spawn step {spawn_step.ep_it_str} with seed {data_order_seed}')
        if models.registry.state_dicts_exist(output_location, self.desc.train_end_step):
            print(f'child already exists')
            return
        print(f'child doesn\'t exist so running train')
        model = models.registry.get(self.desc.model_hparams).to(environment.device())
        train.standard_train(model, output_location, self.desc.dataset_hparams, training_hparams, self._train_location(), spawn_step, save_dense=self.save_dense)
    
    def _average(self, spawn_step, seeds):
        print(f'averaging children for seeds {seeds} at spawn step {spawn_step.ep_it_str}')

        output_location = self._spawn_step_average_location(spawn_step, seeds)
        environment.exists_or_makedirs(output_location)
        spawn_step_location = self._spawn_step_location(spawn_step)
        environment.exists_or_makedirs(spawn_step_location)

        for child_step in self.desc.children_saved_steps(self.save_dense):
            print(f'child step {child_step.ep_it_str}')
            if models.registry.model_exists(output_location, child_step) and is_logger_info_saved(output_location, child_step):
                print('not running average')
                continue
            print('running average')
            standard_average(self.desc.dataset_hparams, self.desc.model_hparams, output_location, spawn_step_location, seeds, child_step)
                
    def run(self, spawn_step_index: int = None):
        print(f'running {self.description()}')

        main_path = self.desc.run_path(part='main', experiment=self.experiment)
        environment.exists_or_makedirs(main_path)
        self.desc.save_hparam(main_path)
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
            indices = [0, len(self.desc.spawn_steps(self.save_dense))]
        for spawn_step_i in range(*indices):
            spawn_step = self.desc.spawn_steps(self.save_dense)[spawn_step_i]
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
    
    def get_w(self, w_name: str):
        """
        get weights from a parent, child, or average model checkpoint

        w_name should be:
            parent_{train_step} for parent
            child_{spawn_step}_{train_step}_{seed}
            avg_{spawn_step}_{train_step}_{seeds}

            where step is in ep_it_str format
            seeds is comma separated
        """
        iterations_per_epoch = datasets.registry.get(self.desc.dataset_hparams).iterations_per_epoch
        w_type = w_name.split('_')[0]
        if w_type == 'parent':
            train_step = Step.from_str(w_name.split('_')[1], iterations_per_epoch)
            return environment.load(paths.model(self._train_location(), train_step))
        elif w_type == 'child':
            params = w_name.split('_')
            spawn_step = Step.from_str(params[1], iterations_per_epoch)
            train_step = Step.from_str(params[2], iterations_per_epoch)
            seed = params[3]
            return environment.load(paths.model(self._spawn_step_child_location(spawn_step, seed), train_step))
        elif w_type == 'avg':
            params = w_name.split('_')
            spawn_step = Step.from_str(params[1], iterations_per_epoch)
            train_step = Step.from_str(params[2], iterations_per_epoch)
            seeds = [i for i in params[3].split(',')]
            return environment.load(paths.model(self._spawn_step_average_location(spawn_step, seeds), train_step))


                

        

