import argparse
from training.runner import TrainingRunner

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    TrainingRunner.add_args(parser)
    args = parser.parse_args()
    TrainingRunner.create_from_args(args).run()

if __name__ == '__main__':
    main()



