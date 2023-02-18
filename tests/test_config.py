import unittest, yaml
from pathlib import Path

from jarvis.config import Config
from jarvis.models.resnet import ResNet


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config_path = Path(__file__).parent/'fixtures'/'resnet_config.yaml'
        self.defaults_path = Path(__file__).parent/'fixtures'/'defaults.yaml'

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

    def test_fill(self):
        config = Config({
            'data': 'CIFAR100',
        })
        config.fill(self.config_path)
        config.lookup(self.defaults_path)
        self.assertEqual(set(config.keys()), set(['data', 'model', 'train']))
        self.assertEqual(config.data, 'CIFAR100')
        self.assertEqual(
            set(config.model.keys()),
            set(['_target_', 'base_channels', 'conv0_channels']),
        )
        self.assertEqual(config.model.base_channels, 64)
        self.assertEqual(config.model.conv0_channels, 32)
        self.assertEqual(
            set(config.train.keys()),
            set(['pretrained', 'batch_size', 'optimizer'])
        )
        self.assertEqual(config.train.optimizer.lr, 1e-4)
        self.assertEqual(config.train.optimizer.momentum, 0.99)

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

    def test_instantiation(self):
        with open(self.config_path, 'r') as f:
            config = Config(yaml.safe_load(f))
        model = config.model.instantiate()
        self.assertIsInstance(model, ResNet)


if __name__=='__main__':
    unittest.main()
