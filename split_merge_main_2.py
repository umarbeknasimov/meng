import argparse
from split_merge.runner2 import SplitMergeRunner2

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    SplitMergeRunner2.add_args(parser)
    args = parser.parse_args()
    SplitMergeRunner2.create_from_args(args).run()

if __name__ == "__main__":
    main()