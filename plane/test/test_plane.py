import torch
import datasets.registry
from environment import environment
from foundations import paths
from foundations.step import Step
import models.registry
from plane import plane
from testing import test_case
from training import train


class TestPlane(test_case.TestCase):
    def setUp(self):
        super(TestPlane, self).setUp()
        self.hparams = models.registry.get_default_hparams('cifar_resnet_20')
        self.hparams.dataset_hparams.subsample_fraction = 0.01 #500, 10
        self.hparams.dataset_hparams.batch_size = 50 #10 iterations per epoch
        self.hparams.dataset_hparams.do_not_augment = True
        self.hparams.training_hparams.data_order_seed = 0
        self.train_loader = datasets.registry.get(self.hparams.dataset_hparams)

        self.hparams.training_hparams.training_steps = '0ep2it'
        self.model = models.registry.get(self.hparams.model_hparams)
        
        self.weights = []
        def callback(output_location, step, model, optimizer, scheduler, logger):
            self.weights.append(self.copy_state(model))

        train.train(self.model, self.hparams.training_hparams, self.train_loader,
                    self.root, callbacks=[callback])
                
        assert len(self.weights) >= 3

    def test_plane_evaluate(self):
        plane.evaluate_plane(self.weights[0], self.weights[1], self.weights[2], self.root, self.hparams.model_hparams, self.hparams.dataset_hparams, 2)

        self.assertTrue(environment.exists(paths.plane_grid(self.root)))
        grid = environment.load(paths.plane_grid(self.root))
        self.assertTrue('x' in grid)
        self.assertTrue('y' in grid)

        self.assertTrue(environment.exists(paths.plane_metrics(self.root)))
        metrics = environment.load(paths.plane_metrics(self.root))
        for metric_i in ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']:
            self.assertTrue(metric_i in metrics)
            self.assertTrue(0 not in metrics[metric_i])




        