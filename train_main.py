import argparse

from foundations.hparams import TrainingHparams, DatasetHparams
from training.desc import TrainingDesc
from training.runner import TrainingRunner

def main(seed, experiment):
    training_hparams = TrainingHparams(data_order_seed=seed)
    dataset_hparams = DatasetHparams()
    training_desc = TrainingDesc(dataset_hparams=dataset_hparams, training_hparams=training_hparams)
    runner = TrainingRunner(training_desc=training_desc, experiment=experiment)
    runner.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    main(args.seed, args.experiment)



