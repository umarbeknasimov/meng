'''
script for computing the loss between parent and 1 child (similar to lookahead)
'''
import argparse
from lookahead import lookahead
from spawning.runner import SpawningRunner

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    SpawningRunner.add_args(parser)
    parser.add_argument('--child_data_order_seed', type=int, required=True)
    args = parser.parse_args()
    spawning_runner = SpawningRunner.create_from_args(args)

    lookahead.compute_lookahead(spawning_runner, args.child_data_order_seed)

if __name__ == "__main__":
    main()
