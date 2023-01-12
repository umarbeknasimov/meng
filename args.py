
from dataclasses import dataclass

@dataclass
class MainArgs:
  epochs: int = 100
  batch_size: int = 128
  lr: float = 0.1
  momentum: float = 0.9
  weight_decay: float = 1e-4
  print_freq: int = 50
  weights_filename: str = 'weights_frankle_seed_1_i=2048_seed_3'
  seed: int = 0