import numpy as np
import datasets
from foundations.step import Step

from testing import test_case
import models.registry
import datasets.registry
from training import train
from training.callbacks import save_state_dicts

class TestTrain(test_case.TestCase):
    def setUp(self):
        super(TestTrain, self).setUp()
        self.hparams = models.registry.get_default_hparams('cifar_resnet_20')
        self.hparams.dataset_hparams.subsample_fraction = 0.01 #500, 10
        self.hparams.dataset_hparams.batch_size = 50 #10 iterations per epoch
        self.hparams.dataset_hparams.do_not_augment = True
        self.hparams.training_hparams.data_order_seed = 0
        self.train_loader = datasets.registry.get(self.hparams.dataset_hparams)

        self.hparams.training_hparams.training_steps = '3ep'
        self.hparams.training_hparams.milestone_steps = '2ep'

        self.model = models.registry.get(self.hparams.model_hparams)
    
        self.step_counter = 0
        self.ep = 0
        self.it = 0
        self.lr = 0.0

        def callback(output_location, step, model, optimizer, scheduler, logger):
            self.step_counter += 1
            self.ep, self.it = step.ep, step.it
            self.lr = np.round(optimizer.param_groups[0]['lr'], 10)
        
        self.callback = callback
    
    def test_train_zero_steps(self):
        before = TestTrain.get_state(self.model)

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    end_step=Step.from_iteration(0, self.train_loader.iterations_per_epoch))

        after = TestTrain.get_state(self.model)

        super(TestTrain, self).assertStateEqual(before, after)
        self.assertEqual(self.step_counter, 0)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 0)
    
    def test_train_two_steps(self):
        before = TestTrain.get_state(self.model)

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    end_step=Step.from_iteration(2, self.train_loader.iterations_per_epoch))

        after = TestTrain.get_state(self.model)

        super(TestTrain, self).assertStateNotEqual(before, after)
        self.assertEqual(self.step_counter, 3)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 2)
        self.assertEqual(self.lr, 0.1)
    
    def test_train_one_epoch(self):
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    end_step=Step.from_epoch(1, 0, self.train_loader.iterations_per_epoch))

        self.assertEqual(self.step_counter, self.train_loader.iterations_per_epoch + 1)
        self.assertEqual(self.ep, 1)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.1)
    
    def test_train_more_than_two_epochs(self):
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    end_step=Step.from_epoch(2, 1, self.train_loader.iterations_per_epoch))

        self.assertEqual(self.step_counter, 2 * self.train_loader.iterations_per_epoch + 2)
        self.assertEqual(self.ep, 2)
        self.assertEqual(self.it, 1)
        self.assertEqual(self.lr, 0.01)
    
    def test_train_in_full(self):
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback])

        self.assertEqual(self.step_counter, 3 * self.train_loader.iterations_per_epoch + 1)
        self.assertEqual(self.ep, 3)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.01)
    
    def test_train_zero_steps_late_start(self):
        before = TestTrain.get_state(self.model)

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    start_step=Step.from_epoch(0, 5, self.train_loader.iterations_per_epoch),
                    end_step=Step.from_epoch(0, 5, self.train_loader.iterations_per_epoch))

        after = TestTrain.get_state(self.model)

        super(TestTrain, self).assertStateEqual(before, after)
        self.assertEqual(self.step_counter, 0)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 0)
    
    def test_train_one_step_late_start(self):
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    start_step=Step.from_epoch(0, 5, self.train_loader.iterations_per_epoch),
                    end_step=Step.from_epoch(0, 6, self.train_loader.iterations_per_epoch))

        self.assertEqual(self.step_counter, 2)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 6)
        self.assertEqual(self.lr, 0.1)
    
    def test_train_one_epoch_late_start(self):
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback], 
                    start_step=Step.from_epoch(0, 5, self.train_loader.iterations_per_epoch),
                    end_step=Step.from_epoch(1, 5, self.train_loader.iterations_per_epoch))

        self.assertEqual(self.step_counter, self.train_loader.iterations_per_epoch + 1)
        self.assertEqual(self.ep, 1)
        self.assertEqual(self.it, 5)
        self.assertEqual(self.lr, 0.1)
    
    def test_train_in_parts(self):
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(0, 5, self.train_loader.iterations_per_epoch))

        self.assertEqual(self.step_counter, 6)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 5)
        self.assertEqual(self.lr, 0.1)

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 5, self.train_loader.iterations_per_epoch),
                    end_step=Step.from_epoch(2, 6, self.train_loader.iterations_per_epoch))

        self.assertEqual(self.step_counter, 2 * self.train_loader.iterations_per_epoch + 6 + 2)
        self.assertEqual(self.ep, 2)
        self.assertEqual(self.it, 6)
        self.assertEqual(self.lr, 0.01)
    
    def test_repeatable_data_order_seed(self):
        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(0, 1, self.train_loader.iterations_per_epoch))
                
        state1 = TestTrain.get_state(self.model)

        self.model.load_state_dict(init)
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(0, 1, self.train_loader.iterations_per_epoch))
        
        state2 = TestTrain.get_state(self.model)

        self.assertStateEqual(state1, state2)
    
    def test_different_data_on_different_epochs(self):
        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(0, 1, self.train_loader.iterations_per_epoch))
                
        state1 = TestTrain.get_state(self.model)

        self.assertStateNotEqual(init, state1)

        self.model.load_state_dict(init)
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, self.train_loader.iterations_per_epoch),
                    end_step=Step.from_epoch(1, 1, self.train_loader.iterations_per_epoch))
        
        state2 = TestTrain.get_state(self.model)

        self.assertStateNotEqual(state1, state2)
    
    def test_train_with_pretrain(self):
        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[save_state_dicts],
                    end_step=Step.from_epoch(0, 1, self.train_loader.iterations_per_epoch))
                
        state1 = TestTrain.get_state(self.model)

        self.assertStateNotEqual(init, state1)

        self.model.load_state_dict(init)
        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[],
                    pretrained_output_location=self.root,
                    pretrained_step=Step.from_epoch(0, 1, self.train_loader.iterations_per_epoch),
                    end_step=Step.from_epoch(0, 0, self.train_loader.iterations_per_epoch))
        
        state2 = TestTrain.get_state(self.model)

        self.assertStateEqual(state1, state2)

    
test_case.main()

