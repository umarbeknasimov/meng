
from dataclasses import dataclass

@dataclass
class TrainingHParams:
  training_steps: str = '100ep'
  batch_size: int = 128
  lr: float = 0.1
  momentum: float = 0.9
  milestone_steps: str = '50ep,75ep'
  weight_decay: float = 1e-4
  seed: int = 0