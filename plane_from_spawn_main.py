'''
script for computing the loss across a plane connecting 3 weights
similar algorithm to garipov
'''
import argparse
import os
from environment import environment
from spawning.runner import SpawningRunner
from plane import plane

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    SpawningRunner.add_args(parser)
    parser.add_argument('--spawn_step_index', type=int, default=None)
    parser.add_argument('--w_1', type=str)
    parser.add_argument('--w_2', type=str)
    parser.add_argument('--w_3', type=str)
    args = parser.parse_args()
    spawning_runner = SpawningRunner.create_from_args(args)


    output_location = os.path.join(
        spawning_runner.desc.run_path(part='planes', experiment=args.experiment),
        '{}|{}|{}'.format(args.w_1, args.w_2, args.w_3))
    
    environment.exists_or_makedirs(output_location)
    
    w_1, w_2, w_3 = spawning_runner.get_w(args.w_1), spawning_runner.get_w(args.w_2), spawning_runner.get_w(args.w_3)

    plane.evaluate_plane(w_1, w_2, w_3, output_location, spawning_runner.desc.model_hparams, spawning_runner.desc.dataset_hparams)

if __name__ == "__main__":
    main()
