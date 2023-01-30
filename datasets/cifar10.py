import torchvision
import numpy as np

from datasets import base

class Dataset(base.ImageDataset):
    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation: bool = True):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = torchvision.datasets.CIFAR10(train=True, root='./data', download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set():
        train_set = torchvision.datasets.CIFAR10(train=True, root='./data', download=True)
        return Dataset(train_set.data, np.array(train_set.targets))

    def __init__(self, examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [], [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


Dataloader = base.DataLoader