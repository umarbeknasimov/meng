from dataclasses import dataclass
import os


from environment import environment
from foundations import desc
from foundations import hparams
from foundations import paths

@dataclass
class TrainingDesc(desc.Desc):
    model_hparams: hparams.ModelHparams
    training_hparams: hparams.TrainingHparams

    def run_path(self):
        return paths.train(
            environment.get_user_dir(), 
            self.training_hparams.seed,
            self.model_hparams.init_step, 
            self.model_hparams.init_step_seed)
    
    def init_state_dict_path(self):
        init_step, init_step_seed = self.model_hparams.init_step, self.model_hparams.init_step_seed
        if init_step and init_step_seed:
            return paths.state_dict(paths.train(environment.get_user_dir(), init_step_seed), init_step)
