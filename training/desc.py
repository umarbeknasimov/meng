from dataclasses import dataclass
import os


from environment import environment
from foundations import desc, hparams, paths
from foundations.step import Step

@dataclass
class TrainingDesc(desc.Desc):
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams

    @staticmethod
    def name_prefix(): return 'train'

    def run_path(self, experiment='main'):
        path = paths.train(os.path.join(
            environment.get_user_dir(),
            experiment, 
            self.hashname))
        if not os.path.exists(path):
            os.makedirs(path)
        return path
