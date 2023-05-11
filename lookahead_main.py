import argparse
from split_merge.lookahead_runner import LookaheadRunner

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    LookaheadRunner.add_args(parser)
    args = parser.parse_args()
    LookaheadRunner.create_from_args(args).run()

if __name__ == "__main__":
    main()