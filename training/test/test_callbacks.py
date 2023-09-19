from foundations.step import Step
from testing import test_case
import math

from training.callbacks import run_at_log_base_2_steps, run_at_log_base_2_steps_dense


class TestCallbacks(test_case.TestCase):
    def setUp(self):
        super(TestCallbacks, self).setUp()
        self.iterations_per_epoch = 50
        self.saved_steps = set()
        def callback(output_location, step, model, optimizer, scheduler, logger, ids_logger):
            self.saved_steps.add(step)
        
        self.callback = callback
    
    def test_run_at_log_base_2_steps(self):
        iterations = 50000
        expected_log_base_2_steps = set({Step.from_iteration(2**i, self.iterations_per_epoch) for i in range(int(math.log2(iterations)) + 1)})
        callback = run_at_log_base_2_steps(self.callback)
        for i in range(iterations):
            callback(None, Step.from_iteration(i, self.iterations_per_epoch), None, None, None, None)
        self.assertSetEqual(self.saved_steps, expected_log_base_2_steps)
    
    def test_run_at_log_base_2_steps_dense(self):
        expected_iterations = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
        expected_log_base_2_steps_dense = set({Step.from_iteration(i, self.iterations_per_epoch) for i in expected_iterations})
        callback = run_at_log_base_2_steps_dense(self.callback)
        for i in range(100):
            callback(None, Step.from_iteration(i, self.iterations_per_epoch), None, None, None, None)
        self.assertSetEqual(self.saved_steps, expected_log_base_2_steps_dense)

test_case.main()