import argparse
from spawning.runner import SpawningRunner

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    SpawningRunner.add_args(parser)
    parser.add_argument('--spawn_step_index', type=int, default=None)
    args = parser.parse_args()
    SpawningRunner.create_from_args(args).run(args.spawn_step_index)

if __name__ == "__main__":
    main()