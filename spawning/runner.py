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
from training.callbacks import standard_ema_callbacks

@dataclass
class SpawningRunner(Runner):
    desc: SpawningDesc
    children_data_order_seeds: list
    experiment: str = 'main'
    ema: bool = False

    @staticmethod
    def description():
        return 'spawn and train children from trained model'
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        parser.add_argument('--ema',  action='store_true')
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
            ema=args.ema)
    
    def _train(self):
        location = self.train_location()
        environment.exists_or_makedirs(location)
        
        if models.registry.state_dicts_exist(location, self.desc.train_end_step): 
            print('train model already exists')
            return

        print('not all spawn steps saved so running train on parent')
        model = models.registry.get(self.desc.model_hparams, self.model_num_classes).to(environment.device())
        if self.desc.pretrain_dataset_hparams and self.desc.pretrain_training_hparams:
            pretrain_output_location = self.pretrain_location()
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
        output_location = self.pretrain_location()
        environment.exists_or_makedirs(output_location)
        if models.registry.state_dicts_exist(output_location, self.desc.pretrain_end_step): 
            print('pretrain model already exists')
            return
        print('pretrain model doesn\'t exist so running pretrain')
    
        model = models.registry.get(self.desc.model_hparams, self.model_num_classes).to(environment.device())
        train.standard_train(model, output_location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams)
    
    def _spawn_and_train(self, spawn_step, data_order_seed):
        training_hparams = TrainingHparams.create_from_instance_and_dict(
            self.desc.training_hparams, {'data_order_seed': data_order_seed})
        output_location = self.spawn_step_child_location(spawn_step, data_order_seed)
        environment.exists_or_makedirs(output_location)
        print(f'child at spawn step {spawn_step.ep_it_str} with seed {data_order_seed}')
        if models.registry.state_dicts_exist(output_location, self.desc.train_end_step):
            print(f'child already exists')
            return
        print(f'child doesn\'t exist so running train')
        model = models.registry.get(self.desc.model_hparams, self.model_num_classes).to(environment.device())
        train.standard_train(model, output_location, self.desc.dataset_hparams, training_hparams, self.train_location(), spawn_step)
    
    def _spawn_and_train_ema(self, spawn_step, data_order_seed):
        print('ema')
        training_hparams = TrainingHparams.create_from_instance_and_dict(
            self.desc.training_hparams, {'data_order_seed': data_order_seed})
        output_location = self.spawn_step_child_location(spawn_step, data_order_seed, part='ema')
        environment.exists_or_makedirs(output_location)
        print(f'child at spawn step {spawn_step.ep_it_str} with seed {data_order_seed}')
        if models.registry.state_dicts_exist(output_location, self.desc.train_end_step):
            print(f'child already exists')
            return
        print(f'child doesn\'t exist so running train')
        model = models.registry.get(self.desc.model_hparams, self.model_num_classes).to(environment.device())
        train_loader = datasets.registry.get(self.desc.dataset_hparams, train=True)
        test_loader = datasets.registry.get(self.desc.dataset_hparams, train=False)
        callbacks = standard_ema_callbacks(
            training_hparams, train_loader, test_loader, model_hparams=self.desc.model_hparams)
        train.train(model, training_hparams, train_loader, 
              output_location, callbacks, pretrained_output_location=self.train_location(), 
              pretrained_step=spawn_step)
    
    def _avg_across(self, parent_step):
        print(f'avg across for {parent_step.ep_it_str}')
        children_steps = self.desc.saved_steps
        avg_location = self.spawn_step_children_location(parent_step, self.children_data_order_seeds, part='avg_across')

        for child_step in children_steps:
            if models.registry.model_exists(avg_location, child_step) and is_logger_info_saved(avg_location, child_step):
                print('not running average')
                continue
            children_weights = []
            children_optimizer_weights = []
            for seed_i in self.children_data_order_seeds:
                child_weights = models.registry.get_model_state_dict(
                    self.spawn_step_child_location(parent_step, seed_i),
                    child_step)
                child_optimizer_weights = models.registry.get_optim_state_dict(
                    self.spawn_step_child_location(parent_step, seed_i),
                    child_step)['optimizer']
                children_weights.append(child_weights)
                children_optimizer_weights.append(child_optimizer_weights)
            standard_average(self.desc.dataset_hparams,
                self.desc.model_hparams,
                self.desc.training_hparams,
                avg_location,
                child_step,
                children_weights,
                children_optimizer_weights)
    
    def _avg_back(self, parent_step):
        print(f'avg back for {parent_step.ep_it_str}')
        children_steps = self.desc.saved_steps
        parent_weights = models.registry.get_model_state_dict(self.train_location(), parent_step)
        parent_optimizer_weights = models.registry.get_optim_state_dict(self.train_location(), parent_step)['optimizer']
        for child_step in children_steps:
            for seed_i in self.children_data_order_seeds:
                avg_location = self.spawn_step_child_location(parent_step, seed_i, part='avg_back')
                if is_logger_info_saved(avg_location, child_step):
                    print('not running average')
                    continue
                child_weights = models.registry.get_model_state_dict(
                    self.spawn_step_child_location(parent_step, seed_i),
                    child_step)
                child_optimizer_weights = models.registry.get_optim_state_dict(
                    self.spawn_step_child_location(parent_step, seed_i),
                    child_step)['optimizer']
                standard_average(self.desc.dataset_hparams,
                    self.desc.model_hparams,
                    self.desc.training_hparams,
                    avg_location,
                    child_step,
                    [child_weights, parent_weights],
                    [child_optimizer_weights, parent_optimizer_weights],
                    dont_save_models=True)
        
    def _avg_back_all(self, parent_step):
        print(f'avg back all for {parent_step.ep_it_str}')
        children_steps = self.desc.saved_steps
        for index, child_step in enumerate(children_steps):
            for seed_i in self.children_data_order_seeds:
                avg_location = self.spawn_step_child_location(parent_step, seed_i, part='avg_back_all')
                if is_logger_info_saved(avg_location, child_step):
                    print('not running average')
                    continue
                all_weights, all_optim_weights = [], []
                for prev_child_step in children_steps[:index + 1]:
                    child_weights = models.registry.get_model_state_dict(
                        self.spawn_step_child_location(parent_step, seed_i),
                        prev_child_step)
                    child_optimizer_weights = models.registry.get_optim_state_dict(
                        self.spawn_step_child_location(parent_step, seed_i),
                        prev_child_step)['optimizer']
                    all_weights.append(child_weights)
                    all_optim_weights.append(child_optimizer_weights)
                standard_average(self.desc.dataset_hparams,
                    self.desc.model_hparams,
                    self.desc.training_hparams,
                    avg_location,
                    child_step,
                    all_weights,
                    all_optim_weights,
                    dont_save_models=True)
    
    def _avg_back_plus_one(self, parent_step):
        print(f'avg back plus one for {parent_step.ep_it_str}')
        iterations_per_epoch = datasets.registry.get_iterations_per_epoch(self.desc.dataset_hparams)
        children_steps = self.desc.saved_steps
        for index, child_step in enumerate(children_steps):
            for seed_i in self.children_data_order_seeds:
                avg_location = self.spawn_step_child_location(parent_step, seed_i, part='avg_back_plus_one')
                if is_logger_info_saved(avg_location, child_step):
                    print('not running average')
                    continue
                all_weights, all_optim_weights = [], []
                steps = [children_steps[0], child_step]
                if index == 2:
                    steps = steps + [children_steps[index - 1]]
                elif index > 2:
                    steps = steps + [children_steps[index - 2]]
                for prev_child_step in steps:
                    child_weights = models.registry.get_model_state_dict(
                        self.spawn_step_child_location(parent_step, seed_i),
                        prev_child_step)
                    child_optimizer_weights = models.registry.get_optim_state_dict(
                        self.spawn_step_child_location(parent_step, seed_i),
                        prev_child_step)['optimizer']
                    all_weights.append(child_weights)
                    all_optim_weights.append(child_optimizer_weights)
                standard_average(self.desc.dataset_hparams,
                    self.desc.model_hparams,
                    self.desc.training_hparams,
                    avg_location,
                    child_step,
                    all_weights,
                    all_optim_weights,
                    dont_save_models=True)
    
    def _avg_back_plus_three(self, parent_step):
        print(f'avg back three one for {parent_step.ep_it_str}')
        iterations_per_epoch = datasets.registry.get_iterations_per_epoch(self.desc.dataset_hparams)
        children_steps = self.desc.saved_steps
        for index, child_step in enumerate(children_steps):
            for seed_i in self.children_data_order_seeds:
                avg_location = self.spawn_step_child_location(parent_step, seed_i, part='avg_back_plus_three')
                if is_logger_info_saved(avg_location, child_step):
                    print('not running average')
                    continue
                all_weights, all_optim_weights = [], []
                steps = [children_steps[0], child_step]
                if index == 2:
                    steps = steps + [children_steps[index - 1]]
                elif index == 3:
                    steps = steps + [children_steps[index - 1], children_steps[index - 2]]
                elif index == 4:
                    steps = steps + [children_steps[index - 1], children_steps[index - 2], children_steps[index - 3]]
                elif index > 3:
                    steps = steps + [children_steps[index - 1], children_steps[index - 2], children_steps[index - 4]]
                
                for prev_child_step in steps:
                    child_weights = models.registry.get_model_state_dict(
                        self.spawn_step_child_location(parent_step, seed_i),
                        prev_child_step)
                    child_optimizer_weights = models.registry.get_optim_state_dict(
                        self.spawn_step_child_location(parent_step, seed_i),
                        prev_child_step)['optimizer']
                    all_weights.append(child_weights)
                    all_optim_weights.append(child_optimizer_weights)
                
                standard_average(self.desc.dataset_hparams,
                    self.desc.model_hparams,
                    self.desc.training_hparams,
                    avg_location,
                    child_step,
                    all_weights,
                    all_optim_weights,
                    dont_save_models=True)
                
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
            indices = [0, len(self.desc.saved_steps)]
        for spawn_step_i in range(*indices):
            spawn_step = self.desc.saved_steps[spawn_step_i]
            for data_order_seed in self.children_data_order_seeds:
                self._spawn_and_train(spawn_step, data_order_seed)
            self._avg_across(spawn_step)
            # self._avg_back(spawn_step)
            # self._avg_back_plus_one(spawn_step)
            self._avg_back_plus_three(spawn_step)
            if self.ema:
               self._spawn_and_train_ema(spawn_step, self.children_data_order_seeds[0])
            # self._avg_back_all(spawn_step)

    def train_location(self):
        return self.desc.run_path(part='parent', experiment=self.experiment)
    
    def pretrain_location(self):
        return self.desc.run_path(part='pretrain', experiment=self.experiment)
    
    def spawn_step_location(self, spawn_step, part='children'):
        return paths.step_(self.desc.run_path(part=part, experiment=self.experiment), spawn_step)
    
    def spawn_step_child_location(self, spawn_step, data_order_seed, part='children'):
        return paths.seed(self.spawn_step_location(spawn_step, part), data_order_seed)
    
    def spawn_step_children_location(self, spawn_step, seeds, part):
        return paths.seeds(self.spawn_step_location(spawn_step, part), seeds)
    
    def get_w(self, w_name: str):
        """
        get weights from a parent, child, or average model checkpoint

        w_name should be:
            parent_{train_step} for parent
            child_{spawn_step}_{train_step}_{seed}

            where step is in ep_it_str format
            seeds is comma separated
        """
        iterations_per_epoch = datasets.registry.get_iterations_per_epoch(self.desc.dataset_hparams)
        w_type = w_name.split('_')[0]
        if w_type == 'parent':
            train_step = Step.from_str(w_name.split('_')[1], iterations_per_epoch)
            return paths.model(self.train_location(), train_step)
        elif w_type == 'child':
            params = w_name.split('_')
            spawn_step = Step.from_str(params[1], iterations_per_epoch)
            train_step = Step.from_str(params[2], iterations_per_epoch)
            seed = params[3]
            return paths.model(self.spawn_step_child_location(spawn_step, seed), train_step)
    
    @property
    def model_num_classes(self):
        return datasets.registry.get(self.desc.dataset_hparams).dataset.num_classes()


                

        
