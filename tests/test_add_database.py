import testsupport.mock_node_environ
from fedbiomed.node.environ import DB_PATH
from fedbiomed.node.data_manager import Data_manager
import unittest
from unittest.mock import MagicMock, patch
import os
import warnings


class TestLoadDataSets(unittest.TestCase):
    """
    Test dataset loading
    Args:
        unittest ([type]): [description]
    """
    
    # Setup data manager
    def setUp(self):
        
        # Ignore ResoruceWarning, this action does not change test results
        warnings.simplefilter('ignore', category=ResourceWarning)
        self.data_manager = Data_manager() 
        pass

    # after the tests
    def tearDown(self): 
        os.remove(DB_PATH)
        pass
    
    def test_load_csv_dataset(self):
        
        """ Test function for loading csv datasets """

        # Load data with header example 
        self.data_manager.add_database( name='test',
                                        tags=['titi'],
                                        data_type='csv',
                                        description='description',
                                        path='tests/test-data/csv/tata-header.csv')

        # Should raise error due to same tag
        with self.assertRaises(Exception):                            
            self.data_manager.add_database( name='test',
                                    tags=['titi'],
                                    data_type='csv',
                                    description='description',
                                    path='tests/test-data/csv/tata-header.csv')

        # Load data with normal different types
        self.data_manager.add_database( name='test',
                                        tags=['tata'],
                                        data_type='csv',
                                        description='description',
                                        path='tests/test-data/csv/titi-normal.csv')

        # Should raise error due to broken csv
        with self.assertRaises(Exception):                            
            self.data_manager.add_database( name='test',
                                    tags=['tutu'],
                                    data_type='csv',
                                    description='description',
                                    path= 'tests/test-data/csv/toto-error.csv')

        pass

    def test_load_image_dataset(self):

        """ Test function for loading image dataset """

        # Load data with header example
        self.data_manager.add_database( name='test',
                                        tags=['titi'],
                                        data_type='images',
                                        description='description',
                                        path='tests/test-data/images')

        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['titi'],
                                            data_type='images',
                                            description='description',
                                            path='tests/test-data/images')
        pass

if __name__ == '__main__':  # pragma: no cover    
    unittest.main() 


