import os
from environment import environment
from foundations.step import Step
from testing import test_case
from training.ids_logger import IdsLogger

class TestIdsLogger(test_case.TestCase):
    def test_create_empty(self):
        IdsLogger()
    
    @staticmethod
    def create_logger():
        logger = IdsLogger()
        logger.add('test_accuracy', Step.from_iteration(0, 400), [1, 2, 3])
        return logger
    
    def test_add_get(self):
        logger = TestIdsLogger.create_logger()
        data = logger.get('test_accuracy', Step.from_iteration(0, 400))
        self.assertIn(1, data)
        self.assertIn(2, data)
        self.assertIn(3, data)
    
    def test_str(self):
        logger = TestIdsLogger.create_logger()
        expected = ['test_accuracy.0.1,2,3']
        self.assertEqual(str(logger), '\n'.join(expected))
    
    def test_create_from_string(self):
        logger = TestIdsLogger.create_logger()
        logger2 = IdsLogger.create_from_str(str(logger))
        self.assertEqual(str(logger), str(logger2))
    
    def test_file_operations(self):
        logger = TestIdsLogger.create_logger()
        save_loc = os.path.join(self.root, 'temp_logger')
        environment.exists_or_makedirs(save_loc)

        logger.save(save_loc)
        logger2 = IdsLogger.create_from_file(save_loc)
        self.assertListEqual(logger.get_data('test_accuracy'), logger2.get_data('test_accuracy'))
        self.assertEqual(str(logger), str(logger2))

test_case.main()