from foundations.hparams import TrainingHparams, DatasetHparams
from training.desc import TrainingDesc
from training.runner import TrainingRunner

def main() :
    pretrain_training_hparams = TrainingHparams()
    pretrain_dataset_hparams = DatasetHparams(transformation_seed=2)
    pretrain_training_desc = TrainingDesc(dataset_hparams=pretrain_dataset_hparams, training_hparams=pretrain_training_hparams)
    pretrain_step = '0ep'

    training_hparams = TrainingHparams()
    dataset_hparams = DatasetHparams(transformation_seed=1)
    training_desc = TrainingDesc(dataset_hparams=dataset_hparams, training_hparams=training_hparams, pretrain_training_desc=pretrain_training_desc, pretrain_step=pretrain_step)

    runner = TrainingRunner(training_desc=training_desc)
    runner.run()

if __name__ == '__main__':
    main()