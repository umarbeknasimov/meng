import abc
from dataclasses import dataclass, fields

from foundations import paths
from foundations.hparams import Hparams

@dataclass
class Desc(abc.ABC):
    """bundle of hyperparams for a kind of job"""
    def save(self, output_location):
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [fields_dict[k].display for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
        with open(paths.hparams(output_location), 'w') as f:
            f.write('\n'.join(hparams_strs))