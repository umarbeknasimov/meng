from dataclasses import dataclass
import os


from environment import environment
from foundations import desc, paths
from training.desc import TrainingDesc
from foundations.hparams import DatasetHparams

@dataclass
class AveragingDesc(desc.Desc):
    train1: TrainingDesc
    train2: TrainingDesc
    dataset_hparams: DatasetHparams

    @staticmethod
    def name_prefix(): return 'avg'

    def run_path(self, get_outer_dir=False, experiment='main'):
        if get_outer_dir:
            return paths.average(os.path.join(
                environment.get_user_dir(),
                experiment))
        return paths.average(os.path.join(
            environment.get_user_dir(),
            experiment, 
            self.hashname))

