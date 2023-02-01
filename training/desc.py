from dataclasses import dataclass
import os


from environment import environment
from foundations import desc, hparams, paths
from foundations.step import Step

@dataclass
class TrainingDesc(desc.Desc):
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    pretrain_training_desc: 'TrainingDesc' = None
    pretrain_step: str = None

    @staticmethod
    def name_prefix(): return 'train'

    def run_path(self, experiment='main'):
        return paths.train(os.path.join(
            environment.get_user_dir(),
            experiment, 
            self.hashname))
