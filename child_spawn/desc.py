from dataclasses import dataclass
import os

from environment import environment
from foundations import desc, hparams, paths
from training.desc import TrainingDesc

@dataclass
class ChildSpawnDesc(desc.Desc):
    parent_training_desc: TrainingDesc
    child_training_desc: TrainingDesc
    spawn_steps: str = 'log'


    @staticmethod
    def name_prefix(): return 'spawn'

    def run_path(self, experiment='main'):
        return paths.train(os.path.join(
            environment.get_user_dir(),
            experiment, 
            self.hashname))
