import os
import shutil
import datasets.registry
from environment import environment
from foundations.step import Step
import models.registry
from split_merge.desc import SplitMergeDesc
from split_merge.runner import SplitMergeRunner
from testing import test_case


class TestSplitMerge(test_case.TestCase):
    def setUp(self):
        super(TestSplitMerge, self).setUp()
        self.hparams = models.registry.get_default_hparams('cifar_resnet_20')
        self.hparams.dataset_hparams.subsample_fraction = 0.01 #500, 10
        self.hparams.dataset_hparams.batch_size = 50 #10 iterations per epoch
        self.hparams.dataset_hparams.do_not_augment = True
        self.hparams.training_hparams.data_order_seed = 0
        self.train_loader = datasets.registry.get(self.hparams.dataset_hparams)

        self.hparams.training_hparams.training_steps = '1it'
        self.iterations_per_epoch = 10
        self.train_end_step = Step.from_str(
            self.hparams.training_hparams.training_steps, self.iterations_per_epoch)

        self.model = models.registry.get(self.hparams.model_hparams)
    
    def test_basic(self):
        desc = SplitMergeDesc(
            training_hparams=self.hparams.training_hparams,
            dataset_hparams=self.hparams.dataset_hparams,
            model_hparams=self.hparams.model_hparams,
        )

        children_seeds = [1, 2]
        num_legs = 2
        runner = SplitMergeRunner(
            desc=desc, 
            children_data_order_seeds=children_seeds,
            num_legs=num_legs)
        
        path = os.path.join(environment.get_user_dir(), 'main', desc.hashname)
        if environment.exists(path):
            print('removing path')
            shutil.rmtree(path)

        runner.run()

        self.assertTrue(environment.exists(runner.leg_location()))
        for leg_i in range(num_legs):
            self.assertTrue(environment.exists(runner.leg_i_location(leg_i)))
            self.assertTrue(environment.exists(runner.parent_location(leg_i)))
            self.assertTrue(environment.exists(
                models.registry.state_dicts_exist(runner.parent_location(leg_i), 
                self.train_end_step)))
            for seed in children_seeds:
                self.assertTrue(environment.exists(
                    models.registry.state_dicts_exist(runner.child_location(leg_i, seed), 
                    self.train_end_step)))
            self.assertTrue(environment.exists(
                models.registry.state_dicts_exist(runner.avg_location(leg_i), 
                self.train_end_step)))
            
            if leg_i != 0:
                parent_optim_state_dict = models.registry.get_optim_state_dict(
                    runner.parent_location(leg_i), 
                    Step.zero(self.iterations_per_epoch)
                )['optimizer']
                 
                parent_model_state_dict = models.registry.get_model_state_dict(
                    runner.parent_location(leg_i),
                    Step.zero(self.iterations_per_epoch)
                )

                avg_optim_state_dict = models.registry.get_optim_state_dict(
                    runner.avg_location(leg_i - 1),
                    self.train_end_step
                )['optimizer']

                avg_model_state_dict = models.registry.get_model_state_dict(
                    runner.avg_location(leg_i - 1),
                    self.train_end_step
                )

                self.assertStateEqual(parent_model_state_dict, avg_model_state_dict)
                # print(parent_optim_state_dict.keys())
                # print(avg_optim_state_dict.keys())
                self.assertOptimizerEqual(parent_optim_state_dict, avg_optim_state_dict, are_dicts=True)
                


        
            


