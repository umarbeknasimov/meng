from dataclasses import dataclass
import os


from environment import environment
from foundations import desc, hparams, paths
from foundations.step import Step
from training.desc import TrainingDesc
from foundations.hparams import DatasetHparams

@dataclass
class AveragingDesc(desc.Desc):
    train1: TrainingDesc
    train2: TrainingDesc
    dataset_hparams: DatasetHparams

    @staticmethod
    def name_prefix(): return 'avg'

    def run_path(self, experiment='main'):
        return paths.train(os.path.join(
            environment.get_user_dir(),
            experiment, 
            self.hashname))
