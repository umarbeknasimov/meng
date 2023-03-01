from foundations.step import Step
from testing import test_case


class StepTest(test_case.TestCase):
    def test_log_2_steps(self):
        iterations_per_epoch = 10
        expected_iterations = [1, 2, 4, 8, 16, 32, 64]
        expected_steps = [Step.from_iteration(i, iterations_per_epoch) for i in expected_iterations]
        end_step = Step.from_iteration(100, iterations_per_epoch)
        steps = Step.get_log_2_steps(end_step, iterations_per_epoch)
        self.assertListEqual(steps, expected_steps)
    def test_log_2_steps_dense(self):
        iterations_per_epoch = 10
        expected_iterations = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
        expected_steps = [Step.from_iteration(i, iterations_per_epoch) for i in expected_iterations]
        end_step = Step.from_iteration(100, iterations_per_epoch)
        steps = Step.get_log_2_steps_dense(end_step, iterations_per_epoch)
        self.assertListEqual(steps, expected_steps)

test_case.main()