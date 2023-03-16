import numpy as np

import datasets.registry
import torch
from testing import test_case
import models.registry

from utils import interpolate


class TestInterpolate(test_case.TestCase):
    def test_average_state_dicts(self):
        weights = [{
            'a': torch.ones((2, 2)),
            'b': torch.zeros((3, 3))
        }, {
            'a': torch.zeros((2, 2)),
            'b': torch.ones((3, 3))
        }, {
            'a': torch.zeros((2, 2)),
            'b': torch.zeros((3, 3))
        }]

        result = interpolate.average_state_dicts(weights)

        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertTrue(np.array_equal(np.ones((2, 2)) * 1/3, result['a'].numpy()))
        self.assertTrue(np.array_equal(np.ones((3, 3)) * 1/3, result['b'].numpy()))
    
    def test_optimizer_average_state_dicts(self):
        param_groups = {
            'lr': 0.1
        }
        optimizers = [{
            'param_groups': param_groups,
            'state': {
                0: {
                    'momentum_1': torch.ones((2, 2)),
                    'momentum_2':  torch.zeros((3, 3))
                }
            }
        }, {
            'param_groups': param_groups,
            'state': {
                0: {
                    'momentum_1': torch.zeros((2, 2)),
                    'momentum_2':  torch.ones((3, 3))
                }
            }
        }, {
            'param_groups': param_groups,
            'state': {
                0: {
                    'momentum_1': torch.zeros((2, 2)),
                    'momentum_2':  torch.zeros((3, 3))
                }
            }
        }]

        result = interpolate.average_optimizer_state_dicts(optimizers)

        self.assertIn('param_groups', result)
        self.assertIn('state', result)
        self.assertIn(0, result['state'])
        self.assertDictEqual(param_groups, result['param_groups'])
        self.assertTrue(np.array_equal(np.ones((2, 2)) * 1/3, result['state'][0]['momentum_1'].numpy()))
        self.assertTrue(np.array_equal(np.ones((3, 3)) * 1/3, result['state'][0]['momentum_2'].numpy()))
    
    def test_forward_pass(self):
        hp = models.registry.get_default_hparams('cifar_resnet_20')
        model = models.registry.get(hp.model_hparams)
        dataloader = datasets.registry.get(hp.dataset_hparams)

        before = self.get_state(model)
        interpolate.forward_pass(model, dataloader)
        after = self.get_state(model)

        for k in after:
            if 'num_batches_tracked' in k or 'mean' in k or 'var' in k:
                self.assertFalse(np.array_equal(before[k], after[k]))
            else:
                self.assertTrue(np.array_equal(before[k], after[k]))


test_case.main()
