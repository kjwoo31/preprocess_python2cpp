import unittest
import os
import shutil
import yaml
from pathlib import Path
from core.mapping.database import MappingDatabase, FunctionMapping

class TestMappingDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary config directory
        self.test_config_dir = Path("test_config")
        self.test_config_dir.mkdir(exist_ok=True)
        (self.test_config_dir / "mappings").mkdir(exist_ok=True)
        (self.test_config_dir / "implementations").mkdir(exist_ok=True)

    def tearDown(self):
        # Clean up
        if self.test_config_dir.exists():
            shutil.rmtree(self.test_config_dir)

    def test_save_learned_mapping_creates_file(self):
        db = MappingDatabase(config_dir=str(self.test_config_dir), auto_load_learned=False)

        mapping = FunctionMapping(
            python_lib="cv2",
            python_func="test_func",
            cpp_lib="cv",
            cpp_func="testFunc",
            cpp_headers=["<opencv2/opencv.hpp>"],
            notes="Test mapping"
        )

        learned_file = self.test_config_dir / "mappings" / "learned.yaml"
        db.save_learned_mapping(mapping, str(learned_file))

        self.assertTrue(learned_file.exists())

        with open(learned_file, 'r') as f:
            data = yaml.safe_load(f)

        self.assertEqual(len(data['functions']), 1)
        self.assertEqual(data['functions'][0]['python_func'], "test_func")
        self.assertEqual(data['functions'][0]['cpp_func'], "testFunc")

    def test_save_learned_mapping_appends(self):
        db = MappingDatabase(config_dir=str(self.test_config_dir), auto_load_learned=False)
        learned_file = self.test_config_dir / "mappings" / "learned.yaml"

        # Create initial file
        initial_data = {
            'functions': [{
                'python_lib': 'cv2',
                'python_func': 'existing_func',
                'cpp_lib': 'cv',
                'cpp_func': 'ExistingFunc',
                'cpp_headers': [],
                'is_method': False,
                'notes': ''
            }]
        }
        with open(learned_file, 'w') as f:
            yaml.dump(initial_data, f)

        mapping = FunctionMapping(
            python_lib="cv2",
            python_func="new_func",
            cpp_lib="cv",
            cpp_func="NewFunc"
        )

        db.save_learned_mapping(mapping, str(learned_file))

        with open(learned_file, 'r') as f:
            data = yaml.safe_load(f)

        self.assertEqual(len(data['functions']), 2)
        self.assertEqual(data['functions'][1]['python_func'], "new_func")

    def test_save_learned_mapping_updates_existing(self):
        db = MappingDatabase(config_dir=str(self.test_config_dir), auto_load_learned=False)
        learned_file = self.test_config_dir / "mappings" / "learned.yaml"

        # Create initial file
        initial_data = {
            'functions': [{
                'python_lib': 'cv2',
                'python_func': 'update_func',
                'cpp_lib': 'cv',
                'cpp_func': 'OldFunc',
                'cpp_headers': [],
                'is_method': False,
                'notes': ''
            }]
        }
        with open(learned_file, 'w') as f:
            yaml.dump(initial_data, f)

        mapping = FunctionMapping(
            python_lib="cv2",
            python_func="update_func",
            cpp_lib="cv",
            cpp_func="NewFunc",
            notes="Updated"
        )

        db.save_learned_mapping(mapping, str(learned_file))

        with open(learned_file, 'r') as f:
            data = yaml.safe_load(f)

        self.assertEqual(len(data['functions']), 1)
        self.assertEqual(data['functions'][0]['cpp_func'], "NewFunc")
        self.assertIn("Updated", data['functions'][0]['notes'])

if __name__ == '__main__':
    unittest.main()
