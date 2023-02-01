import abc

import torch
import torchvision
import numpy as np

class Dataset(abc.ABC, torch.utils.data.Dataset):
    """base class for all datasets"""

    @staticmethod
    @abc.abstractmethod
    def num_test_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_train_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_classes() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_train_set(use_agumentation: bool) -> 'Dataset':
        pass

    @staticmethod
    @abc.abstractmethod
    def get_test_set() -> 'Dataset':
        pass

    
    def __init__(self, examples: np.ndarray, labels: np.ndarray):
        """create dataset object
        
        examples: numpy array of examples
        labels: numpy array of labels

        """

        if examples.shape[0] != labels.shape[0]:
            raise ValueError('different number of examples ({}) and labels ({})'.format(examples.shape[0], labels.shape[0]))
        
        self._examples = examples
        self._labels = labels
        self._subsampled = False
    
    def randomize_labels(self, seed: int, fraction: float) -> None:
        """randomize labels of specified fraction of the dataset"""

        num_to_randomize = np.ceil(self.num_train_examples() * fraction).astype(int)
        randomized_labels = np.random.RandomState(seed=seed).randint(self.num_classes(), size=num_to_randomize)
        examples_to_randomize = np.random.RandomState(seed=seed+1).permutation(np.arange(len(self._labels)))[:num_to_randomize]
        self._labels[examples_to_randomize] = randomized_labels
    
    def subsample(self, seed: int, fraction: float):
        if self._subsampled:
            raise ValueError('cannot subsample more than once')
        
        num_to_subsample = np.ceil(self.num_train_examples() * fraction).astype(int)
        examples_in_subsample = np.random.RandomState(seed=seed).randint(self.num_classes(), size=num_to_subsample)
        self._examples, self_labels = self._examples[examples_in_subsample], self_labels[examples_in_subsample]

    def __len__(self):
        return self._labels.size
    
    def __getitem__(self, index):
        return self._examples[index], self._labels[index]

class ImageDataset(Dataset):
    @abc.abstractmethod
    def example_to_img(self, example: np.ndarray): pass

    def __init__(self, examples, labels, image_transforms=None, tensor_transforms=None):
        super(ImageDataset, self).__init__(examples, labels)
        self._image_transforms = image_transforms or []
        self._tensor_transforms = tensor_transforms or []

        self._composed = torchvision.transforms.Compose(self._image_transforms + [torchvision.transforms.ToTensor()] + self._tensor_transforms)
    
    def __getitem__(self, index):
        example, label = self._examples[index], self._labels[index]
        example = self._composed(self.example_to_img(example))
        return example, label

class ShuffleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_examples):
        self._num_examples = num_examples
        self._seed = None
    
    def __iter__(self):
        if self._seed == -1:
            indices = list(range(self._num_samples))
        elif self._seed is None:
            indices = torch.randperm(self._num_examples).tolist()
        else:
            g = torch.Generator()
            g.manual_seed(self._seed)
            indices = torch.randperm(self._num_examples, generator=g).tolist()
        return iter(indices)
    
    def __len__(self):
        return self._num_examples
    
    def shuffle_dataorder(self, seed: int):
        self._seed = seed

class DataLoader(torch.utils.data.DataLoader):
    """wrapper to access custom shuffling logic"""

    def __init__(self, dataset: Dataset, batch_size: int):
        self._sampler = ShuffleSampler(len(dataset))
        self._iterations_per_epoch = np.ceil(len(dataset) / batch_size).astype(int)
        super(DataLoader, self).__init__(
            dataset, batch_size, sampler=self._sampler
        )
    
    def shuffle(self, seed: int):
        self._sampler.shuffle_dataorder(seed)
    
    @property
    def iterations_per_epoch(self):
        return self._iterations_per_epoch
