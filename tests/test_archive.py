import unittest, yaml
from pathlib import Path
import random, shutil
import numpy as np

from jarvis.config import Config
from jarvis.archive import Archive, HashableRecordArchive, ConfigArchive


class ArchiveTestCase(unittest.TestCase):

    def _create_axv(self, store_dir):
        raise NotImplementedError

    def setUp(self):
        self.tmp_dir = 'test_'+''.join([
            '{:X}'.format(random.choice(range(16))) for _ in range(4)
        ])
        self.axv = self._create_axv(store_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


class TestBasicArchive(ArchiveTestCase):

    def _create_axv(self, store_dir):
        return Archive(store_dir)

    def test_basic_dict_ops(self):
        self.axv._clear()
        # check invalid key
        key = '0'
        with self.assertRaises(KeyError):
            self.axv[key] = 2.718
        # check value assignment
        key = '00000000'
        with self.assertRaises(KeyError):
            self.axv[key]
        self.axv[key] = 3.14
        self.assertEqual(len(self.axv), 1)
        self.assertIn(key, self.axv)
        self.assertEqual(self.axv[key], 3.14)
        # check deletion
        del self.axv[key]
        self.assertNotIn(key, self.axv)


class TestHashableRecordArchive(ArchiveTestCase):

    def _create_axv(self, store_dir):
        return HashableRecordArchive(store_dir)

    def test_array(self):
        self.axv._clear()
        x = np.random.uniform(size=(3, 4, 5))
        key = self.axv.add(x)
        y = self.axv[key]
        self.assertEqual(y.shape, (3, 4, 5))
        self.assertEqual(np.all(y==x), True)


class TestConfigArchive(ArchiveTestCase):

    def _create_axv(self, store_dir):
        return ConfigArchive(store_dir)

    def setUp(self):
        super().setUp()
        self.config_path = Path(__file__).parent/'fixtures'/'resnet_config.yaml'

    def test_adding_config(self):
        self.axv._clear()
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        # check adding a config
        key = self.axv.add(config)
        self.assertIn(key, self.axv)
        self.assertIsInstance(self.axv[key], Config)
        self.assertEqual(self.axv.add(config), key)
        self.assertEqual(len(self.axv), 1)
        # checking adding a new config
        config['xx'] = 'new_val'
        self.assertNotEqual(self.axv.add(config), key)
        self.assertEqual(len(self.axv), 2)
        # check duplicate detection
        key = self.axv._new_key()
        self.axv[key] = config
        dups = self.axv.get_duplicates()
        self.assertEqual(len(dups), 1)
        self.assertEqual(len(dups[0][1]), 2)


if __name__=='__main__':
    unittest.main()
