"""
script for evaluation the children spawned
"""

from constants import *
import dataset
from utils import load
import evaluate
import models
import json
import evaluate

children_dir = USER_DIR + '/children'

eval_file = children_dir + '/eval.json'
stats_i = {
        'train': {
            # (seed1, seed2, avg) tuple
            'loss': [],
            'acc': []
        },
        'valid': {
            'loss': [],
            'acc': []
        }
    }

max_i = 13

result = []
train_loader, val_loader = dataset.get_train_val_loaders()

# train loss, eval loss for every i, every k
for i in range(max_i + 1):
    seed1_weights = load.load(f'{children_dir}/seed1_i={i}', DEVICE)
    seed2_weights = load.load(f'{children_dir}/seed2_i={i}', DEVICE)
    assert(len(seed1_weights) == len(seed2_weights))
    num_weights = len(seed1_weights)
    curr_stats = stats_i.copy()
    for k in range(num_weights):
        data_type = 'train'
        for data in [train_loader, val_loader]:
            seed1_model = models.frankleResnet20().to(DEVICE)
            seed1_model.load_state_dict(seed1_weights[k])
            seed1_loss, seed1_acc = evaluate.evaluate_data_loader(seed1_model, data, DEVICE)

            seed2_model = models.frankleResnet20().to(DEVICE)
            seed2_model.load_state_dict(seed2_weights[k])
            seed2_loss, seed2_acc = evaluate.evaluate_data_loader(seed2_model, train_loader, DEVICE)

            avg_loss, avg_acc = evaluate.eval_interpolation(seed1_weights[k], seed2_weights[k], 0.5, data, DEVICE, True)
            loss = [seed1_loss, seed2_loss, avg_loss]
            acc = [seed1_acc, seed2_acc, avg_acc]
            print('loss values ', loss)
            print('acc values ', acc)
            if data_type == 'train':
                curr_stats[data_type]['loss'].append(loss)
                curr_stats[data_type]['acc'].append(acc)
                data_type = 'valid'
    result.append(curr_stats)
    with open(eval_file, 'w') as f:
        json_string = json.dumps(result)
        json.dump(json_string, f)
        print('saving results file')

        


