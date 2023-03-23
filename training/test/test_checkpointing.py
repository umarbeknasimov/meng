import datasets.registry
from environment import environment
from foundations import paths
from foundations.step import Step
import models.registry
from testing import test_case
from training import callbacks, checkpointing, optimizers
from training.metric_logger import MetricLogger


class TestCheckpointing(test_case.TestCase):
    def test_create_restore(self):
        hp = models.registry.get_default_hparams('cifar_resnet_20')
        hp.training_hparams.milestone_steps = '80ep,120ep'
        model = models.registry.get(hp.model_hparams)
        optimizer = optimizers.get_optimizer(model, hp.training_hparams)
        scheduler = optimizers.get_lr_scheduler(hp.training_hparams, 400, optimizer)
        dataloader = datasets.registry.get(hp.dataset_hparams)
        step = Step.from_epoch(13, 27, 400)

        examples, labels = next(iter(dataloader))
        optimizer.zero_grad()
        model.train()
        model.loss_criterion(model(examples), labels).backward()
        optimizer.step()
        scheduler.step()

        logger = MetricLogger()
        logger.add('test_accuracy', Step.from_epoch(0, 0, 400), 0.1)
        logger.add('test_accuracy', Step.from_epoch(10, 0, 400), 0.5)
        logger.add('test_accuracy', Step.from_epoch(100, 0, 400), 0.8)

        checkpointing.save_checkpoint_callback(self.root, step, model, optimizer, scheduler, logger)
        self.assertTrue(environment.exists(paths.checkpoint(self.root)))

        # new model
        model2 = models.registry.get(hp.model_hparams)
        optimizer2 = optimizers.get_optimizer(model, hp.training_hparams)
        scheduler2 = optimizers.get_lr_scheduler(hp.training_hparams, 400, optimizer)

        self.assertStateNotEqual(model.state_dict(), model2.state_dict())
        self.assertIn('momentum_buffer', optimizer.state[optimizer.param_groups[0]['params'][0]])
        self.assertNotIn('momentum_buffer', optimizer2.state[optimizer.param_groups[0]['params'][0]])
        self.assertNotEqual(scheduler.state_dict()['_step_count'], scheduler2.state_dict()['_step_count'])

        step2, logger2 = checkpointing.restore_checkpoint(self.root, model2, optimizer2, scheduler2, 400)

        self.assertTrue(environment.exists(paths.checkpoint(self.root)))
        self.assertEqual(step, step2)
        self.assertEqual(str(logger), str(logger2))

        self.assertStateEqual(model.state_dict(), model2.state_dict())
        self.assertOptimizerEqual(optimizer.state_dict(), optimizer2.state_dict())
        self.assertSchedulerEqual(scheduler, scheduler2)
    
    def test_create_restore_no_scheduler(self):
        hp = models.registry.get_default_hparams('cifar_resnet_20')
        model = models.registry.get(hp.model_hparams)
        optimizer = optimizers.get_optimizer(model, hp.training_hparams)
        dataloader = datasets.registry.get(hp.dataset_hparams)
        step = Step.from_epoch(13, 27, 400)

        examples, labels = next(iter(dataloader))
        optimizer.zero_grad()
        model.train()
        model.loss_criterion(model(examples), labels).backward()
        optimizer.step()

        logger = MetricLogger()
        logger.add('test_accuracy', Step.from_epoch(0, 0, 400), 0.1)

        checkpointing.save_checkpoint_callback(self.root, step, model, optimizer, None, logger)
        self.assertTrue(environment.exists(paths.checkpoint(self.root)))

        # new model
        model2 = models.registry.get(hp.model_hparams)
        optimizer2 = optimizers.get_optimizer(model, hp.training_hparams)

        step2, logger2 = checkpointing.restore_checkpoint(self.root, model2, optimizer2, None, 400)

        self.assertTrue(environment.exists(paths.checkpoint(self.root)))
        self.assertEqual(step, step2)
        self.assertEqual(str(logger), str(logger2))

        self.assertStateEqual(model.state_dict(), model2.state_dict())
        self.assertOptimizerEqual(optimizer.state_dict(), optimizer2.state_dict())
    
    def test_load_pretrained(self):
        hp = models.registry.get_default_hparams('cifar_resnet_20')
        model = models.registry.get(hp.model_hparams)
        optimizer = optimizers.get_optimizer(model, hp.training_hparams)
        dataloader = iter(datasets.registry.get(hp.dataset_hparams))
        step = Step.from_epoch(13, 27, 400)

        examples, labels = next(dataloader)
        optimizer.zero_grad()
        model.train()
        model.loss_criterion(model(examples), labels).backward()
        optimizer.step()

        callbacks.save_state_dicts(self.root, step, model, optimizer, None, None)
        self.assertTrue(models.registry.state_dicts_exist(self.root, step))

        # new model
        model2 = models.registry.get(hp.model_hparams)
        optimizer2 = optimizers.get_optimizer(model2, hp.training_hparams)

        examples2, labels2 = next(dataloader)
        optimizer2.zero_grad()
        model2.train()
        model2.loss_criterion(model2(examples2), labels2).backward()
        optimizer2.step()

        checkpointing.load_pretrained(self.root, step, model2, optimizer2, None, True)

        self.assertStateEqual(model.state_dict(), model2.state_dict())
        self.assertOptimizerNotEqual(optimizer.state_dict(), optimizer2.state_dict())

        model3 = models.registry.get(hp.model_hparams)
        optimizer3 = optimizers.get_optimizer(model3, hp.training_hparams)

        checkpointing.load_pretrained(self.root, step, model3, optimizer3, None, False)
        self.assertStateEqual(model.state_dict(), model3.state_dict())
        self.assertOptimizerEqual(optimizer.state_dict(), optimizer3.state_dict())

test_case.main()