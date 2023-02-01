import torch

from foundations.hparams import TrainingHparams, DatasetHparams
from training.desc import TrainingDesc
from training.runner import TrainingRunner

def main():

    training_hparams = TrainingHparams(data_order_seed=1)
    dataset_hparams = DatasetHparams()
    training_desc = TrainingDesc(dataset_hparams=dataset_hparams, training_hparams=training_hparams)
    runner = TrainingRunner(training_desc=training_desc)
    runner.run()

if __name__ == '__main__':
    main()



