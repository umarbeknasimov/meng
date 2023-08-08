import numpy as np
import torch
from environment import environment
from foundations import paths
from spawning.runner import SpawningRunner
from utils.euclidean import l2_distance, norm
import itertools

from utils.state_dict import flatten_state_dict_w_o_batch_stats

def cos_sim(state1, state2):
    return torch.cosine_similarity(
        flatten_state_dict_w_o_batch_stats(state1), flatten_state_dict_w_o_batch_stats(state2), dim=0)


def compute_cos_sim(spawning_runner: SpawningRunner):
    output_location = spawning_runner.desc.run_path(part='cos_sim', experiment=spawning_runner.experiment)
    environment.exists_or_makedirs(output_location)
    steps = spawning_runner.desc.saved_steps
    metrics_shape = (len(steps), len(steps))

    if environment.exists(paths.distances_metrics(output_location)):
        metrics = torch.load(paths.distances_metrics(output_location))
    else:
        metrics = {
            'between_children': np.zeros(metrics_shape),
            'between_child_parent': np.zeros(metrics_shape)
        }

    for i, spawn_step in enumerate(steps):
        parent_model = environment.load(paths.model(
                spawning_runner.train_location(), spawn_step))
        for j, child_step in enumerate(steps):
            print(f'{i}, {j}')
            if metrics['between_children'][i][j] != 0 and metrics['between_child_parent'][i][j] != 0 and metrics['child'][i][j] != 0:
                print(' skipping')
                continue
            between_children_d = 0
            between_child_parent = 0
            for seed_index in range(len(spawning_runner.children_data_order_seeds)):
                seed = spawning_runner.children_data_order_seeds[seed_index]
                seed_model = environment.load(paths.model(
                        spawning_runner.spawn_step_child_location(
                            spawn_step, seed), child_step))
                between_child_parent += cos_sim(seed_model, parent_model)
            
            seed_combs = set(itertools.combinations(spawning_runner.children_data_order_seeds, 2))
            for seed0, seed1 in seed_combs:
                    seed0_model = environment.load(paths.model(
                        spawning_runner.spawn_step_child_location(
                            spawn_step, seed0), child_step))
                    seed1_model = environment.load(paths.model(
                        spawning_runner.spawn_step_child_location(
                            spawn_step, seed1), child_step))
                    between_children_d += cos_sim(seed0_model, seed1_model)

            metrics['between_children'][i][j] = between_children_d / len(seed_combs)
            metrics['between_child_parent'][i][j] = between_child_parent / len(spawning_runner.children_data_order_seeds)

            print(f' between children : {metrics["between_children"][i][j]}')
            print(f' between child, parent : {metrics["between_child_parent"][i][j]}')

            torch.save(metrics, paths.distances_metrics(output_location))