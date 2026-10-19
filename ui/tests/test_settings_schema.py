"""Validate runner configurations, including invalid settings that would fail at runtime."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

CONFIGS = Path(__file__).resolve().parents[2] / 'configs'


class SettingsSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((CONFIGS / 'schema.json').read_text())
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)
        cls.md = json.loads((CONFIGS / 'example_workflow_NVE.json').read_text())
        cls.fire = json.loads((CONFIGS / 'example_workflow_Fire.json').read_text())

    def test_all_existing_workflows(self):
        for path in sorted(CONFIGS.glob('example_workflow_*.json')):
            with self.subTest(path=path.name):
                self.validator.validate(json.loads(path.read_text()))

    def test_missing_required_runtime_values_are_rejected(self):
        paths = [
            ('meta', 'name'), ('common_settings', 'atoms', 'symbol'),
            ('common_settings', 'atoms', 'ratio'),
            ('common_settings', 'interactions', 'neighbour_list', 'cutoff'),
            ('common_settings', 'interactions', 'potentials', 'epsilon'),
            ('steps', 0, 'name'), ('steps', 0, 'observer'),
            ('steps', 0, 'simulation', 'dt'),
            ('steps', 0, 'simulation', 'ensemble', 'temperature'),
        ]
        for path in paths:
            with self.subTest(path=path):
                data = deepcopy(self.md)
                current = data
                for key in path[:-1]:
                    current = current[key]
                del current[path[-1]]
                self.assertFalse(self.validator.is_valid(data))

    def test_invalid_types_bounds_and_binary_pair_lengths(self):
        cases = [
            (('common_settings', 'atoms', 'n_atoms'), 0),
            (('common_settings', 'atoms', 'density'), 0),
            (('common_settings', 'atoms', 'ratio'), [0, 0]),
            (('common_settings', 'atoms', 'ratio'), ['0.8', '0.2']),
            (('common_settings', 'atoms', 'symbol'), ['X']),
            (('common_settings', 'interactions', 'neighbour_list', 'cutoff'), [1.5, 2, 2]),
            (('common_settings', 'interactions', 'potentials', 'sigma'), [1]),
            (('common_settings', 'interactions', 'neighbour_list', 'max_neighbours'), 0),
            (('steps', 0, 'simulation', 'dt'), 0),
            (('steps', 0, 'simulation', 'log_step'), 0),
            (('steps', 0, 'simulation', 'ensemble', 'tau'), 0),
            (('steps', 0, 'simulation', 'ensemble', 'seed'), -1),
            (('steps', 0, 'simulation', 'ensemble', 'type'), 'NPT'),
            (('steps', 0, 'observer', 'divisions'), 0),
            (('steps',), []),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                data = deepcopy(self.md)
                current = data
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = value
                self.assertFalse(self.validator.is_valid(data))

    def test_all_thermostats_and_temperature_schedulers(self):
        for thermostat in ('Bussi', 'Nose-Hoover', 'Langevin'):
            for scheduler in ('constant', 'linear'):
                with self.subTest(thermostat=thermostat, scheduler=scheduler):
                    data = deepcopy(self.md)
                    ensemble = data['steps'][0]['simulation']['ensemble']
                    ensemble.update(thermostat=thermostat, scheduler=scheduler)
                    if thermostat == 'Nose-Hoover':
                        ensemble.pop('seed')
                    else:
                        ensemble['seed'] = 18446744073709551615
                    if scheduler == 'linear':
                        self.assertFalse(self.validator.is_valid(data))
                        ensemble['rate_per_unit_time'] = -0.01
                    self.validator.validate(data)

    def test_observer_required_values_for_each_output_type(self):
        examples = {
            'linear': {'interval': 1}, 'log': {'divisions': 9},
            'linear_trajectory': {'interval': 1, 'is_unwrap': True},
            'log_trajectory': {'divisions': 9, 'is_unwrap': False},
        }
        for kind, fields in examples.items():
            with self.subTest(kind=kind):
                data = deepcopy(self.md)
                data['steps'][0]['observer'] = dict(type=kind, **fields)
                self.validator.validate(data)
                for field in fields:
                    missing = deepcopy(data)
                    del missing['steps'][0]['observer'][field]
                    self.assertFalse(self.validator.is_valid(missing))

    def test_fire_defaults_custom_parameters_and_exclusive_step_kind(self):
        self.validator.validate(self.fire)
        data = deepcopy(self.fire)
        data['steps'][1]['minimize']['params'] = 'default'
        self.validator.validate(data)
        data['steps'][1]['minimize']['params'] = {'n_max': 100}
        self.assertFalse(self.validator.is_valid(data))
        data = deepcopy(self.fire)
        data['steps'][1]['minimize']['params']['dt_start'] = 0
        self.assertFalse(self.validator.is_valid(data))
        data = deepcopy(self.fire)
        data['steps'][1]['simulation'] = deepcopy(data['steps'][0]['simulation'])
        self.assertFalse(self.validator.is_valid(data))
        data = deepcopy(self.fire)
        del data['steps'][1]['minimize']
        self.assertFalse(self.validator.is_valid(data))

    def test_runner_fallbacks_allow_omitting_optional_values(self):
        data = deepcopy(self.md)
        data['meta'].pop('unit')
        data['meta'].pop('seed')
        interactions = data['common_settings']['interactions']
        interactions.pop('use_graph')
        interactions.pop('cell_list')
        interactions['neighbour_list'].pop('margin')
        interactions['neighbour_list'].pop('max_neighbours')
        interactions['potentials'].pop('type')
        ensemble = data['steps'][0]['simulation']['ensemble']
        ensemble.pop('thermostat')  # Nose-Hoover is the runner's fallback.
        ensemble.pop('scheduler')
        ensemble.pop('tau')
        ensemble.pop('seed')
        data['steps'][0]['simulation'].pop('log_step')
        data['steps'][0]['observer'] = {'interval': 100}  # linear fallback
        data['steps'][1]['simulation']['ensemble'].pop('type')  # NVE fallback
        self.validator.validate(data)


if __name__ == '__main__':
    unittest.main()
