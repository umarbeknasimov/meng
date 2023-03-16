import argparse
from split_merge.runner import SplitMergeRunner

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    SplitMergeRunner.add_args(parser)
    args = parser.parse_args()
    SplitMergeRunner.create_from_args(args).run()

if __name__ == "__main__":
    main()