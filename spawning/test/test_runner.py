import os
import shutil

from environment import environment
from foundations import paths
from foundations.step import Step
from spawning.desc import SpawningDesc
from spawning.runner import SpawningRunner
from testing import test_case
import models.registry
import datasets.registry
from training import optimizers


class TestRunner(test_case.TestCase):
    def setUp(self):
        super(TestRunner, self).setUp()
        self.hparams = models.registry.get_default_hparams('cifar_resnet_20')
        self.hparams.dataset_hparams.subsample_fraction = 0.01 #500, 10
        self.hparams.dataset_hparams.batch_size = 50 #10 iterations per epoch
        self.hparams.dataset_hparams.do_not_augment = True
        self.hparams.training_hparams.data_order_seed = 0
        self.train_loader = datasets.registry.get(self.hparams.dataset_hparams)

        self.hparams.training_hparams.training_steps = '0ep3it'
        self.hparams.training_hparams.milestone_steps = '0ep1it'

        self.model = models.registry.get(self.hparams.model_hparams)
    
    # def test_no_pretrain_no_average(self):
    #     desc = SpawningDesc(
    #         training_hparams=self.hparams.training_hparams,
    #         dataset_hparams=self.hparams.dataset_hparams,
    #         model_hparams=self.hparams.model_hparams,
    #     )
    #     # single seed, so no averaging
    #     children_seeds = [1]
    #     runner = SpawningRunner(desc=desc, children_data_order_seeds=children_seeds)
    #     path = os.path.join(environment.get_user_dir(), 'main', desc.hashname)
    #     if environment.exists(path):
    #         print('removing path')
    #         shutil.rmtree(path)

    #     runner.run()
        
    #     self.assertFalse(environment.exists(runner.pretrain_location()))
    #     self.assertTrue(environment.exists(runner.train_location()))

    #     for spawn_step in desc.saved_steps:
    #         child_model_train_path = runner.spawn_step_child_location(spawn_step, children_seeds[0])
    #         parent_model_train_path = runner.train_location()

    #         self.assertTrue(environment.exists(runner.spawn_step_location(spawn_step)))
    #         self.assertFalse(environment.exists(runner.spawn_step_average_location(spawn_step, children_seeds)))
    #         self.assertTrue(environment.exists(child_model_train_path))
    #         # child training
    #         for child_step in desc.saved_steps:
    #             self.assertTrue(environment.exists(paths.model(runner.spawn_step_child_location(spawn_step, children_seeds[0]), child_step)))
            
    #         # check if state at parent spawn step is same as child step 0
    #         child_model_step_0 = environment.load(paths.model(child_model_train_path, Step.zero(10)))
    #         parent_model_spawn_step = environment.load(paths.model(parent_model_train_path, spawn_step))
    #         self.assertStateEqual(child_model_step_0, parent_model_spawn_step)

    #         child_optim_step_0 = environment.load(paths.optim(child_model_train_path, Step.zero(10)))
    #         parent_optim_spawn_step = environment.load(paths.optim(parent_model_train_path, spawn_step))
            
    #         child_scheduler_step_0_state, child_optimizer_step_0_state = child_optim_step_0['scheduler'], child_optim_step_0['optimizer']
    #         parent_scheduler_spawn_step_state, parent_optimizer_spawn_step_state = parent_optim_spawn_step['scheduler'], parent_optim_spawn_step['optimizer']

    #         child_optimizer_step_0 = optimizers.get_optimizer(self.model, self.hparams.training_hparams)
    #         child_optimizer_step_0.load_state_dict(child_optimizer_step_0_state)
    #         parent_optimizer_spawn_step = optimizers.get_optimizer(self.model, self.hparams.training_hparams)
    #         parent_optimizer_spawn_step.load_state_dict(parent_optimizer_spawn_step_state)

    #         child_scheduler_step_0 = optimizers.get_lr_scheduler(self.hparams.training_hparams, 10, child_optimizer_step_0)
    #         child_scheduler_step_0.load_state_dict(child_scheduler_step_0_state)
    #         parent_scheduler_spawn_step = optimizers.get_lr_scheduler(self.hparams.training_hparams, 10, parent_optimizer_spawn_step)
    #         parent_scheduler_spawn_step.load_state_dict(parent_scheduler_spawn_step_state)

    #         self.assertOptimizerEqual(child_optimizer_step_0, parent_optimizer_spawn_step)
    #         self.assertSchedulerEqual(child_scheduler_step_0, parent_scheduler_spawn_step)
    
    def test_average(self):
        desc = SpawningDesc(
            training_hparams=self.hparams.training_hparams,
            dataset_hparams=self.hparams.dataset_hparams,
            model_hparams=self.hparams.model_hparams,
        )
        children_seeds = [1, 2]
        runner = SpawningRunner(desc=desc, children_data_order_seeds=children_seeds)
        path = os.path.join(environment.get_user_dir(), 'main', desc.hashname)
        if environment.exists(path):
            print('removing path')
            shutil.rmtree(path)

        runner.run()
        for spawn_step in desc.saved_steps:
            average_spawn_step_path = runner.spawn_step_average_location(spawn_step, children_seeds)
            self.assertTrue(environment.exists(average_spawn_step_path))
            for child_step in desc.saved_steps:
                self.assertTrue(environment.exists(paths.model(average_spawn_step_path, child_step)))
    
    def test_pretrain(self):
        desc = SpawningDesc(
            training_hparams=self.hparams.training_hparams,
            dataset_hparams=self.hparams.dataset_hparams,
            model_hparams=self.hparams.model_hparams,
            pretrain_dataset_hparams=self.hparams.dataset_hparams,
            pretrain_training_hparams=self.hparams.training_hparams
        )

        children_seeds = [1]
        runner = SpawningRunner(desc=desc, children_data_order_seeds=children_seeds)
        path = os.path.join(environment.get_user_dir(), 'main', desc.hashname)
        if environment.exists(path):
            print('removing path')
            shutil.rmtree(path)
        
        runner.run()
        
        pretrain_path = runner.pretrain_location()
        self.assertTrue(environment.exists(pretrain_path))
        self.assertTrue(environment.exists(
            paths.model(pretrain_path,  Step.from_str(self.hparams.training_hparams.training_steps, 10))))
        
        pretrain_model_end_step = environment.load(paths.model(pretrain_path, Step.from_str(self.hparams.training_hparams.training_steps, 10)))
        train_model_zero_step = environment.load(paths.model(runner.train_location(), Step.zero(10)))

        self.assertStateEqual(pretrain_model_end_step, train_model_zero_step)

    def test_spawn_step_index(self):
        desc = SpawningDesc(
            training_hparams=self.hparams.training_hparams,
            dataset_hparams=self.hparams.dataset_hparams,
            model_hparams=self.hparams.model_hparams,
        )
        children_seeds = [1, 2]
        runner = SpawningRunner(desc=desc, children_data_order_seeds=children_seeds)
        path = os.path.join(environment.get_user_dir(), 'main', desc.hashname)
        if environment.exists(path):
            print('removing path')
            shutil.rmtree(path)

        # spawn_step_index 2 = step 0ep2it
        runner.run(2)
        critical_spawn_step = Step.from_epoch(0, 2, 10)

        parent_model_train_path = runner.train_location()
        self.assertTrue(environment.exists(parent_model_train_path))

        for spawn_step in desc.saved_steps:
            self.assertTrue(environment.exists(paths.model(parent_model_train_path, spawn_step)))
            self.assertTrue(environment.exists(paths.optim(parent_model_train_path, spawn_step)))
            if spawn_step == critical_spawn_step:
                self.assertTrue(environment.exists(runner.spawn_step_location(spawn_step)))
                for seed in children_seeds:
                    child_model_train_path = runner.spawn_step_child_location(spawn_step, seed)
                    self.assertTrue(environment.exists(child_model_train_path))
                self.assertTrue(environment.exists(runner.spawn_step_average_location(spawn_step, children_seeds)))
            else:
                self.assertFalse(environment.exists(runner.spawn_step_location(spawn_step)))
    
    def test_get_w(self):
        self.hparams.training_hparams.training_steps = '0ep1it'
        self.hparams.training_hparams.milestone_steps = None

        desc = SpawningDesc(
            training_hparams=self.hparams.training_hparams,
            dataset_hparams=self.hparams.dataset_hparams,
            model_hparams=self.hparams.model_hparams,
        )
        children_seeds = [1, 2]
        runner = SpawningRunner(desc=desc, children_data_order_seeds=children_seeds)
        path = os.path.join(environment.get_user_dir(), 'main', desc.hashname)
        if environment.exists(path):
            print('removing path')
            shutil.rmtree(path)

        runner.run()
        for spawn_step in desc.saved_steps:
            parent_w = f'parent_{spawn_step.ep_it_str}'
            self.assertStateEqual(environment.load(runner.get_w(parent_w)), environment.load(paths.model(runner.train_location(), spawn_step)))
            average_spawn_step_path = runner.spawn_step_average_location(spawn_step, children_seeds)
            for child_step in desc.saved_steps:
                for child_seed in children_seeds:
                    child_w = f'child_{spawn_step.ep_it_str}_{child_step.ep_it_str}_{child_seed}'
                    self.assertTrue(environment.load(runner.get_w(child_w)), environment.load(paths.model(runner.spawn_step_child_location(spawn_step, child_seed), child_step)))
                avg_w = f'avg_{spawn_step.ep_it_str}_{child_step.ep_it_str}_{",".join([str(i) for i in children_seeds])}'
                self.assertTrue(environment.load(runner.get_w(avg_w)), environment.load((paths.model(average_spawn_step_path, child_step))))
        

        
test_case.main()