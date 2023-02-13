from dataclasses import dataclass
import datasets.registry

from environment import environment
from foundations.runner import Runner
import models.registry
from training.callbacks import run_every_epoch, save_state_dicts, standard_callbacks
from training.train import train
from training.desc import TrainingDesc

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    save_every_epoch: bool = False
    experiment: str = 'main'
    

    @staticmethod
    def description():
        return 'train a model'
    
    def run(self):
        print(f'running {self.description()}')

        train_loader = datasets.registry.get(self.training_desc.dataset_hparams)
        test_loader = datasets.registry.get(self.training_desc.dataset_hparams, False)
        callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader)
        if self.save_every_epoch:
            callbacks.append(run_every_epoch(save_state_dicts))

        model = models.registry.get(self.training_desc.model_hparams).to(environment.device())

        output_location = self.training_desc.run_path(self.experiment)
        environment.exists_or_makedirs(output_location)
        self.training_desc.save_hparam(output_location)

        train(model, self.training_desc.training_hparams, train_loader, output_location, callbacks)