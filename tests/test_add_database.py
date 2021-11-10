# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Detele environ. It is necessary to rebuild environ for required component
delete_environ()
import testsupport.mock_common_environ
# Import environ for node since test will be runing for node component
from fedbiomed.node.environ    import environ


from fedbiomed.node.data_manager import Data_manager
import unittest
import os
import warnings

import inspect

print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class TestLoadDataSets(unittest.TestCase):
    """
    Test dataset loading
    Args:
        unittest ([type]): [description]
    """

    # Setup data manager
    def setUp(self):

        self.testdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                ),
            "test-data"
            )

        # Ignore ResoruceWarning, this action does not change test results
        warnings.simplefilter('ignore', category=ResourceWarning)
        self.data_manager = Data_manager()
        pass

    # after the tests
    def tearDown(self):
        os.remove(environ['DB_PATH'])
        pass

    def test_load_csv_dataset(self):

        """ Test function for loading csv datasets """

        # Load data with header example
        self.data_manager.add_database( name='test',
                                        tags=['titi'],
                                        data_type='csv',
                                        description='description',
                                        path=os.path.join( self.testdir,
                                                           "csv",
                                                           "tata-header.csv"
                                                          )
                                        )

        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['titi'],
                                            data_type='csv',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                               "csv",
                                                               "tata-header.csv"
                                                              )
                                           )

        # Load data with normal different types
        self.data_manager.add_database( name='test',
                                        tags=['tata'],
                                        data_type='csv',
                                        description='description',
                                        path=os.path.join( self.testdir,
                                                           "csv",
                                                           "titi-normal.csv"
                                                          )
                                       )

        # Should raise error due to broken csv
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['tutu'],
                                            data_type='csv',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                               "csv",
                                                               "toto-error.csv"
                                                              )
                                           )
            pass

    def test_load_image_dataset(self):

        """ Test function for loading image dataset """

        # Load data with header example
        self.data_manager.add_database( name='test',
                                        tags=['titi'],
                                        data_type='images',
                                        description='description',
                                        path=os.path.join( self.testdir,
                                                           "images"
                                                          )
                                       )

        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['titi'],
                                            data_type='images',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                               "images"
                                                              )
                                           )
            pass

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
