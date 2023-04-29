import os
import sys
import torchvision
import numpy as np
from PIL import Image

from datasets import base
from environment import environment

class CIFAR100(torchvision.datasets.CIFAR100):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.
    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        original_stdout = sys.stdout
        with open(os.devnull, mode='w') as fp:
            sys.stdout = fp
            super(CIFAR100, self).download()
            sys.stdout = original_stdout

class Dataset(base.ImageDataset):
    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_classes(): return 100

    @staticmethod
    def get_train_set(use_augmentation: bool = True):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR100(train=True, root='./data', download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set():
        train_set = CIFAR100(train=False, root='./data', download=True)
        return Dataset(train_set.data, np.array(train_set.targets))

    def __init__(self, examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [], [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_img(self, example):
        return Image.fromarray(example)


Dataloader = base.DataLoader