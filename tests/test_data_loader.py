"""
Unit tests for data loader module
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary CSV file for testing
        self.test_data = pd.DataFrame({
            'Time': [0, 1, 2, 3, 4],
            'V1': [1.1, 2.2, 3.3, 4.4, 5.5],
            'V2': [-1.1, -2.2, -3.3, -4.4, -5.5],
            'Amount': [100, 200, 300, 400, 500],
            'Class': [0, 0, 1, 0, 1]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        self.data_loader = DataLoader(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)
    
    def test_load_data_success(self):
        """Test successful data loading"""
        df = self.data_loader.load_data()
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ['Time', 'V1', 'V2', 'Amount', 'Class'])
    
    def test_load_data_file_not_found(self):
        """Test file not found error"""
        loader = DataLoader("nonexistent_file.csv")
        with self.assertRaises(FileNotFoundError):
            loader.load_data()
    
    def test_get_data_info(self):
        """Test data info extraction"""
        df = self.data_loader.load_data()
        info = self.data_loader.get_data_info(df)
        
        self.assertEqual(info['shape'], (5, 5))
        self.assertEqual(info['class_distribution'], {0: 3, 1: 2})
        self.assertIn('Time', info['columns'])
    
    def test_split_features_target(self):
        """Test feature-target splitting"""
        df = self.data_loader.load_data()
        X, y = self.data_loader.split_features_target(df)
        
        self.assertEqual(len(X.columns), 4)  # Should exclude 'Class'
        self.assertNotIn('Class', X.columns)
        self.assertEqual(len(y), 5)
        self.assertEqual(y.name, 'Class')

if __name__ == '__main__':
    unittest.main()