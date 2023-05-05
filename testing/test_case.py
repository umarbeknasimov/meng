import os
import shutil
import unittest
import numpy as np

from environment import environment


class TestCase(unittest.TestCase):
    def setUp(self):
        root = os.path.join(environment.get_user_dir(), 'TESTING')
        if environment.exists(root):
            shutil.rmtree(root)
        environment.makedirs(root)
        self.root = root

    def tearDown(self):
        if environment.exists(self.root):
            shutil.rmtree(self.root)
    
    @staticmethod
    def get_state(model):
        """return copy of state of model detached and converted to numpy"""

        return {k: v.clone().detach().cpu().numpy() for k, v in model.state_dict().items()}
    
    @staticmethod
    def copy_state(model):
        """return copy of state of model"""

        return {k: v.clone() for k, v in model.state_dict().items()}
        
    def assertStateEqual(self, state1, state2):
        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertTrue(np.array_equal(state1[k], state2[k]), f'key {k}')
    
    def assertStateNotEqual(self, state1, state2):
        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            if 'num_batches_tracked' in k:
                continue
            self.assertFalse(np.array_equal(state1[k], state2[k]), f'key {k}')
    
    def assertOptimizerEqual(self, optimizer1, optimizer2, are_dicts=False):
        if are_dicts:
            optim1_state = optimizer1
            optim2_state = optimizer2
        else:
            optim1_state = optimizer1.state_dict()
            optim2_state = optimizer2.state_dict()
        # print(optim1_state)
        if len(optim1_state['state'].keys()) == 0: return
        for param_key in optim1_state['state'].keys(): # checking for param_name, in our case is just 'momentum_buffer'
            for param_name in optim1_state['state'][param_key]:
                self.assertTrue(np.array_equal(optim1_state['state'][param_key][param_name].numpy(), optim2_state['state'][param_key][param_name].numpy()))
    
    def assertOptimizerNotEqual(self, optimizer1, optimizer2):
        if 'momentum_buffer' not in optimizer1['state'][optimizer1['param_groups'][0]['params'][0]]:
            self.assertNotIn('momentum_buffer', optimizer2['state'][optimizer2['param_groups'][0]['params'][0]])
            return
        mom1 = optimizer1['state'][optimizer1['param_groups'][0]['params'][0]]['momentum_buffer']
        mom2 = optimizer2['state'][optimizer2['param_groups'][0]['params'][0]]['momentum_buffer']
        self.assertFalse(np.array_equal(mom1.numpy(), mom2.numpy()))
    
    def assertSchedulerEqual(self, scheduler1, scheduler2):
        self.assertEqual(scheduler1.state_dict()['_step_count'], scheduler2.state_dict()['_step_count'])

def main():
    if __name__ == '__main__':
        unittest.main()