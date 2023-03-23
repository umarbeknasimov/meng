from dataclasses import asdict, fields
from testing import test_case
from foundations import hparams


class TestHParams(test_case.TestCase):
    def test_create_from_instance_and_dict(self):
        default_hparams = hparams.TrainingHparams()
        derived_hparams = hparams.Hparams.create_from_instance_and_dict(
            default_hparams, {'data_order_seed': 2})
        
        default_hparams_dict = asdict(default_hparams)
        derived_hparams_dict = asdict(derived_hparams)
        
        for field in fields(default_hparams):
            if field.name == 'data_order_seed':
                self.assertEqual(default_hparams_dict[field.name], None)
                self.assertEqual(derived_hparams_dict[field.name], 2)
            else:
                self.assertEqual(default_hparams_dict[field.name], derived_hparams_dict[field.name])



