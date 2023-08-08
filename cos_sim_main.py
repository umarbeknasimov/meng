'''
script for computing the distances between children & between parent and child
'''
import argparse
from distances.cos_sim import compute_cos_sim
from spawning.runner import SpawningRunner

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    SpawningRunner.add_args(parser)
    args = parser.parse_args()
    spawning_runner = SpawningRunner.create_from_args(args)
    compute_cos_sim(spawning_runner)
    

if __name__ == "__main__":
    main()