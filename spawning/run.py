import os

from dataclasses import dataclass, asdict
from datasets import registry
from environment import environment
from foundations.runner import Runner
from foundations.hparams import TrainingHparams
from foundations import paths
from models.cifar_resnet import Model
from spawning.desc import SpawningDesc
from training import train
from training import pre_trained
from training.metric_logger import MetricLogger
from training.callbacks import create_eval_callback, save_logger, save_model
from utils import interpolate, state_dict

@dataclass
class SpawningRunner(Runner):
    desc: SpawningDesc
    children_data_order_seeds: str

    @staticmethod
    def description():
        return 'spawn and train children from trained model'
    
    def _train(self):
        location = self.desc.run_path('parent')
        environment.exists_or_makedirs(location)
        
        if os.path.exists(paths.state_dict(location, self.desc.train_end_step)): 
            print('train model already exists')
            return

        print('not all spawn steps saved so running train on parent')
        model = Model().to(environment.device())
        if self.desc.pretrain_dataset_hparams and self.desc.pretrain_training_hparams:
            pretrain_output_location = self.desc.run_path('pretrain')
            train.standard_train(
                model, location, self.desc.dataset_hparams, 
                self.desc.training_hparams, pretrain_output_location, 
                self.desc.pretrain_end_step)
        else:
            train.standard_train(
                model, location, self.desc.dataset_hparams, 
                self.desc.training_hparams)
    
    def _pretrain(self):
        if self.desc.pretrain_dataset_hparams and self.desc.pretrain_training_hparams:
            print('pretraning args exist')
            location = self.desc.run_path('pretrain')
            environment.exists_or_makedirs(location)
            if os.path.exists(paths.state_dict(location, self.desc.pretrain_end_step)): 
                print('pretrain model already exists')
                return

            print('pretrain model doesn\'t exist so running pretrain')
        
            model = Model().to(environment.device())
            train.standard_train(model, location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams)
    
    def _spawn_and_train(self, spawn_step, data_order_seed):
        modified_parent_training_hparams = asdict(self.desc.training_hparams)
        modified_parent_training_hparams['data_order_seed'] = data_order_seed
        print('modified parent training hparams ', modified_parent_training_hparams)
        training_hparams=TrainingHparams.create_from_dict(modified_parent_training_hparams)
        output_location = paths.spawn_instance(self.desc.run_path('children'), spawn_step, data_order_seed)
        environment.exists_or_makedirs(output_location)
        print(f'child at spawn step {spawn_step.ep}ep{spawn_step.it}it with seed {data_order_seed}')
        if os.path.exists(paths.state_dict(output_location, self.desc.train_end_step)): 
            print(f'child already exists')
            return
        print(f'child doesn\'t exist so running train')
        model = Model().to(environment.device())
        train.standard_train(model, output_location, self.desc.dataset_hparams, training_hparams, self.desc.run_path('parent'), spawn_step)
    
    def _average(self, spawn_step, seeds):
        print(f'averaging children for seeds {seeds} at spawn step {spawn_step.ep}ep{spawn_step.it}it')
        train_loader = registry.get(self.desc.dataset_hparams)
        test_loader = registry.get(self.desc.dataset_hparams, False)
        test_eval_callback = create_eval_callback('test', test_loader)
        train_eval_callback = create_eval_callback('train', train_loader)
        callbacks = [test_eval_callback, train_eval_callback, save_logger, save_model]
        logger = MetricLogger()
        output_location = paths.spawn_average(self.desc.run_path('children'), spawn_step, seeds)
        environment.exists_or_makedirs(output_location)
        for child_step in self.desc.children_saved_steps:
            weights = []
            for data_order_seed in seeds:
                weights.append(
                    pre_trained.get_pretrained_model_state_dict(
                        paths.spawn_instance(self.desc.run_path('children'), spawn_step, data_order_seed), 
                        child_step))
            model = Model().to(environment.device())
            averaged_weights = interpolate.average_state_dicts(weights)
            averaged_weights_wo_batch_stats = state_dict.get_state_dict_wo_batch_stats(averaged_weights)
            model.load_state_dict(averaged_weights_wo_batch_stats)
            interpolate.forward_pass(model, train_loader)
            for callback in callbacks: callback(
                output_location,
                child_step,
                model,
                None,
                None,
                logger)
            
        
    def run(self):
        print(f'running {self.description()}')

        self.desc.save_hparam(self.desc.run_path())
        
        self._pretrain()
        self._train()
        
        seeds = [int(seed) for seed in self.children_data_order_seeds.split(',')]

        print(f'spawning children with seeds {seeds}')

        for spawn_step in self.desc.spawn_steps:
            for data_order_seed in seeds:
                self._spawn_and_train(spawn_step, data_order_seed)
            self._average(spawn_step, seeds)


                

        

