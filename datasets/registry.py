from datasets import cifar10, cifar100
from foundations.hparams import DatasetHparams

registered_datasets = {'cifar10': cifar10, 'cifar100': cifar100}

def get(dataset_hparams: DatasetHparams, train: bool = True):
    """get train or test set corresponding to dataset hparams"""
    seed = dataset_hparams.transformation_seed or 0

    #dataset
    if dataset_hparams.dataset_name in registered_datasets:
        use_augmentation = train and not dataset_hparams.do_not_augment
        if train:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation)
        else:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_test_set()
    else:
        raise ValueError('no such dataset {}'.format(dataset_hparams.dataset_name))

    #transform
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed, dataset_hparams.random_labels_fraction)

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed, dataset_hparams.subsample_fraction)

    #loader
    return registered_datasets[dataset_hparams.dataset_name].Dataloader(dataset, dataset_hparams.batch_size)

def get_iterations_per_epoch(dataset_hparams: DatasetHparams):
    return get(dataset_hparams).iterations_per_epoch
