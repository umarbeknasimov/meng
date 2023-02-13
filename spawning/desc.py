import os

from dataclasses import dataclass
from foundations import desc
from foundations.hparams import ModelHparams, TrainingHparams, DatasetHparams
from foundations.step import Step
import datasets.registry
from foundations import desc, paths
from environment import environment

@dataclass
class SpawningDesc(desc.Desc):
    training_hparams: TrainingHparams
    dataset_hparams: DatasetHparams
    model_hparams: ModelHparams
    pretrain_training_hparams: TrainingHparams = None
    pretrain_dataset_hparams: DatasetHparams = None
    

    @staticmethod
    def name_prefix(): return 'spawn'

    def run_path(self, part='main'):
        path = os.path.join(
            environment.get_user_dir(), 
            self.hashname,
            part)
        environment.exists_or_makedirs(path)
        return path
    
    def str_to_step(self, s: str, pretrain: bool = False) -> Step:
        dataset_hparams = self.pretrain_dataset_hparams if pretrain else self.dataset_hparams
        iterations_per_epoch = datasets.registry.get(dataset_hparams).iterations_per_epoch
        return Step.from_str(s, iterations_per_epoch)
    
    @property
    def pretrain_end_step(self):
        return self.str_to_step(self.pretrain_training_hparams.training_steps, True)
    
    @property
    def train_end_step(self):
        return self.str_to_step(self.training_hparams.training_steps)
    
    def _train_dataset_log2_steps(self):
        iterations_per_epoch = datasets.registry.get(self.dataset_hparams).iterations_per_epoch
        return Step.get_log_2_steps(self.train_end_step, iterations_per_epoch)
    
    @property
    def spawn_steps(self):
        return self._train_dataset_log2_steps()
    
    @property
    def children_saved_steps(self):
        return self._train_dataset_log2_steps()

    

    



