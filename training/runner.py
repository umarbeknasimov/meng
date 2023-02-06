from dataclasses import dataclass

from environment import environment
from foundations.runner import Runner
from models.cifar_resnet import Model
from training.train import standard_train
from training.desc import TrainingDesc

@dataclass
class TrainingRunner(Runner):
    training_desc: TrainingDesc
    experiment: str = 'main'
    verbose: bool = True

    @staticmethod
    def description():
        return 'train a model'
    
    def run(self):
        if self.verbose:
            print(f'running {self.description()}')

        # train_loader = registry.get(self.training_desc.dataset_hparams)
        # test_loader = registry.get(self.training_desc.dataset_hparams, False)
        # callbacks = standard_callbacks(self.training_desc.training_hparams, train_loader, test_loader)

        model = Model().to(environment.device())

        output_location = self.training_desc.run_path(self.experiment)
        environment.exists_or_makedirs(output_location)
        self.training_desc.save_hparam(output_location)

        standard_train(
            model, 
            output_location,
            self.training_desc.dataset_hparams,
            self.training_desc.training_hparams)


