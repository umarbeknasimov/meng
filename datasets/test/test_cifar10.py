import numpy as np

from datasets import cifar10
from testing import test_case


class TestDataset(test_case.TestCase):
    def setUp(self):
        super(TestDataset, self).setUp()
        self.test_set = cifar10.Dataset.get_test_set()
        self.train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        self.train_set_noaugment = cifar10.Dataset.get_train_set(use_augmentation=False)

    def test_not_none(self):
        self.assertIsNotNone(self.test_set)
        self.assertIsNotNone(self.train_set)
        self.assertIsNotNone(self.train_set_noaugment)
    
    def test_size(self):
        self.assertEqual(cifar10.Dataset.num_classes(), 10)
        self.assertEqual(cifar10.Dataset.num_train_examples(), 50000)
        self.assertEqual(cifar10.Dataset.num_test_examples(), 10000)
    
    def test_randomize_labels_half(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0.5)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        print(examples_match)
        self.assertEqual(examples_match, 5503)
    
    def test_randomize_labels_none(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 10000)
    
    def test_randomize_labels_all(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 1)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 1020)

    def test_subsample(self):
        labels_test = [3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
        subsammpled_labels_with_seed_zero_test = [6, 3, 0, 0, 6, 1, 0, 6, 8, 6]

        self.assertEqual(self.test_set._labels[:10].tolist(), labels_test)
        self.test_set.subsample(0, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsammpled_labels_with_seed_zero_test)

        subsammpled_labels_with_seed_one_test = [6, 3, 1, 6, 3, 3, 8, 6, 1, 1]
        self.test_set = cifar10.Dataset.get_test_set()
        self.test_set.subsample(1, 0.1)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsammpled_labels_with_seed_one_test)

test_case.main()