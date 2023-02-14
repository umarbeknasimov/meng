import argparse

from training.runner import TrainingRunner

# def main(seed, experiment, save_every_epoch):
#     training_hparams = TrainingHparams(data_order_seed=seed)
#     dataset_hparams = DatasetHparams()
#     training_desc = TrainingDesc(dataset_hparams=dataset_hparams, training_hparams=training_hparams)
#     runner = TrainingRunner(training_desc=training_desc, experiment=experiment, save_every_epoch=save_every_epoch)
#     runner.run()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int)
#     parser.add_argument('--experiment', type=str)
#     parser.add_argument('--save_every_epoch', type=bool, default=False)
#     args = parser.parse_args()
#     main(args.seed, args.experiment, args.save_every_epoch)


def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    TrainingRunner.add_args(parser)
    args = parser.parse_args()
    TrainingRunner.create_from_args(args).run()

if __name__ == '__main__':
    main()



