import abc
from dataclasses import dataclass, fields
import hashlib
import os

from environment import environment
from foundations import paths
from foundations.hparams import Hparams

@dataclass
class Desc(abc.ABC):
    """bundle of hyperparams for a kind of job"""

    def save_hparam(self, local_output_location):
        with open(paths.hparams(local_output_location), 'w') as f:
            f.write(str(self))
        
    @property
    def hashname(self) -> str:
        """The name under which experiments with these hyperparameters will be stored."""

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict)]
        hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
        return f'{self.name_prefix()}_{hash_str}'
    
    def __str__(self) -> str:
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = []
        for k in sorted(fields_dict):
            hparams_strs.append(k)
            if isinstance(fields_dict[k], Hparams):
                hparams_strs.append(fields_dict[k].display)
            elif isinstance(fields_dict[k], Desc):
                hparams_strs.append(fields_dict[k].run_path())
            elif fields_dict[k] is not None:
                hparams_strs.append(fields_dict[k])
        hparams_strs.append(self.hashname)
        return '\n'.join(hparams_strs)
    
