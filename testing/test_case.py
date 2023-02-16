import os
import shutil
import unittest
import numpy as np

from environment import environment


class TestCase(unittest.TestCase):
    def setUp(self):
        testing_path = os.path.join(environment.get_user_dir(), 'TESTING')
        if environment.exists(testing_path):
            shutil.rmtree(testing_path)
        environment.makedirs(testing_path)
        self.testing_path = testing_path

    def tearDown(self):
        if environment.exists(self.testing_path):
            shutil.rmtree(self.testing_path)
    
    @staticmethod
    def get_state(model):
        """return copy of state of model"""

        return {k: v.clone().detach().cpu().numpy() for k, v in model.state_dict.items()}
        
    def assertStateEqual(self, state1, state2):
        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertTrue(np.array_equal(state1[k], state2[k]))
    
    def assertStateNotEqual(self, state1, state2):
        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertFalse(np.array_equal(state1[k], state2[k]))

def main():
    if __name__ == '__main__':
        unittest.main()