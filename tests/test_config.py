import unittest, yaml
from pathlib import Path

from jarvis.config import Config


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config_path = Path(__file__).parent/'fixtures/resnet.yaml'

    def test_nesting(self):
        config = Config({
            'a.b': 1,
            'a.c': True,
        })
        # check nesting structure
        self.assertEqual(set(config.keys()), set(['a']))
        self.assertEqual(set(config['a'].keys()), set(['b', 'c']))
        self.assertEqual(config['a']['b'], 1)
        self.assertEqual(config['a.b'], 1)
        # check value assignment
        config['a.d'] = -3.14
        self.assertEqual(set(config['a'].keys()), set(['b', 'c', 'd']))
        self.assertEqual(config['a']['d'], -3.14)
        config['a.d'] = 2.718
        self.assertEqual(config['a']['d'], 2.718)
        # check update method
        config.update({'b.a': 'John'})
        self.assertEqual(set(config.keys()), set(['a', 'b']))
        self.assertEqual(config['b']['a'], 'John')

    def test_dot_notation(self):
        with open(self.config_path, 'r') as f:
            config = Config(yaml.safe_load(f))
        # test read
        self.assertIsInstance(config.train.optimizer, Config)
        # test write
        config.data = 'CIFAR100'
        config.train.optimizer.weight_decay = 1e-4
        self.assertEqual(config.data, 'CIFAR100')
        self.assertEqual(config.train.optimizer.weight_decay, 1e-4)


if __name__=='__main__':
    unittest.main()
