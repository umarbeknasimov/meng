import os

import torch
from foundations.hparams import TrainingHparams, ModelHparams
from training.desc import TrainingDesc
from training.runner import TrainingRunner

import ssl
def main() :
    ssl._create_default_https_context = ssl._create_unverified_context
    training_hparams = TrainingHparams(seed=2)
    model_hparams = ModelHparams()
    training_desc = TrainingDesc(model_hparams=model_hparams, training_hparams=training_hparams)
    runner = TrainingRunner(training_desc=training_desc)
    runner.run()

if __name__ == '__main__':
    main()



