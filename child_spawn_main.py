"""
generates and saves children models based on some model's training run

let's say a model has iterates: W_0, W_1, ..., W_f, where W_i is iteration i
we usually keep track of W_0, W_1, W_2, W_4, W_8, ... W_{2**i} for space efficiency
for each W, generate 2 copies (children) with seed 1 and seed 2
train each child with learning rate 0.1 for 4096 iterations (10 epochs)
save the weights of each child (every 2**i iterations)

the goal of this:
 to analyze the mode connectivity of the children through various i
 to find critical i
"""

from utils import load
import main
import torch
import models
from args import MainArgs
from constants import *

parent_file = USER_DIR + "/models/weights_frankle_seed_1"
children_dir = USER_DIR + "/children"

parent_weights = load.load(parent_file, DEVICE)

seed1Args = MainArgs(epochs=10, seed=1)
seed2Args = MainArgs(epochs=10, seed=2)


for i in range(14, len(parent_weights)):
    # need to change weights filename
    seed1Args.weights_filename = f'{children_dir}/seed1_i={i}'
    seed2Args.weights_filename = f'{children_dir}/seed2_i={i}'

    torch.manual_seed(seed1Args.seed)
    torch.cuda.manual_seed(seed1Args.seed)
    
    seed1 = models.frankleResnet20().to(DEVICE)
    seed1.load_state_dict(parent_weights[i])
    main.main(seed1, seed1Args, DEVICE)

    torch.manual_seed(seed2Args.seed)
    torch.cuda.manual_seed(seed2Args.seed)

    seed2 = models.frankleResnet20().to(DEVICE)
    seed2.load_state_dict(parent_weights[i])
    main.main(seed2, seed2Args, DEVICE)


    

