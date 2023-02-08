import argparse
from foundations.hparams import DatasetHparams, TrainingHparams
from spawning.run import SpawningRunner
from spawning.desc import SpawningDesc

def main(parent_seed, children_seeds, spawn_step_index):
    training_hparams = TrainingHparams(data_order_seed=parent_seed)
    dataset_hparams = DatasetHparams()

    pretrain_training_hparams = TrainingHparams()
    pretrain_dataset_hparams = DatasetHparams(random_labels_fraction=1.0)

    spawning_desc = SpawningDesc(
        training_hparams=training_hparams,
        dataset_hparams=dataset_hparams,
        pretrain_dataset_hparams=pretrain_dataset_hparams,
        pretrain_training_hparams=pretrain_training_hparams
    )
    
    spawning_runner = SpawningRunner(
        desc=spawning_desc,
        children_data_order_seeds=children_seeds
    )

    spawning_runner.run(spawn_step_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_seed', type=int)
    parser.add_argument('--children_seeds', type=str)
    parser.add_argument('--spawn_step_index', type=int, default=None)
    args = parser.parse_args()
    main(args.parent_seed, args.children_seeds, args.spawn_step_index)