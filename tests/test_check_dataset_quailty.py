from fedbiomed.researcher.job import Job
import unittest

from unittest.mock import MagicMock, patch


class TestJobDatasetQualityCheck(unittest.TestCase):
    """
    Test `check_dataset_qualty` method of Job class
    Args:
        unittest ([type]): [description]
    """
    
    # Setup HistoryMonitor with Mocking messaging 
    def setUp(self):
        pass

    # after the tests
    def tearDown(self):
        pass
    
    def test_check_quality_csv(self):
        
        """Test when federated csv datasets do match for federated training"""


        ## Test when every thing is okay
        try: 
            job_cllient_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}]
            }
            Job.check_data_quality(job_cllient_datasets)

        except:
            self.assertTrue(False, 'Raised an error when datasets have good quality')

        # Check when there are more than 2 datasets
        try:
            job_cllient_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}]
            } 
            Job.check_data_quality(job_cllient_datasets)
        
        except:
             self.assertTrue(False, 'Raised an error when there are more than 3 \
                  datasets even they have good quality')


        # When csv datasets has diffrent number of rows
        try:
            job_cllient_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [100,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [15,5]}]
            }  
            Job.check_data_quality(job_cllient_datasets)
        
        except:
             self.assertTrue(False, 'Raised an error when datasets have different number \
                                        of rows even they have good quality')


        # Dimension error when the dimensions are different
        with self.assertRaises(Exception):
            job_cllient_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,16]}]
            }

            Job.check_data_quality(job_cllient_datasets)
                
        # Check when the variable datatypes are differents
        with self.assertRaises(Exception):
            job_cllient_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'int' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'string'], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,6]}]
            }

            Job.check_data_quality(job_cllient_datasets)

        # Check when the datatypes are different
        with self.assertRaises(Exception):
            job_cllient_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,6]}]
            }

            Job.check_data_quality(job_cllient_datasets)
    
    def test_check_quality_image(self):
        
        """Test when federated image dataset do match with good qualty"""

        ## Test when everything is okay
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error when datasets have good quality')


        ## Test when there are more than 2 datasets 
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error when datasets have good quality')
    

        ## Test when the number of samples are different
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,3, 10, 10]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 10, 10]}]
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error when datasets have different sample sizes')
    

        # Check when color channels are different
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,1, 10, 10]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 10, 10]}]
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error while checking color channels')

        # Check when image shapes are different
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,3, 10, 25]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 10, 5]}]
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error while checking image shapes')
        
        # Check when the images gets transposed
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 20, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,3, 10, 20]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 20, 10]}]
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error while checking transposed images')


        # Check when image color channels and shapes are transposed
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 20, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,1, 10, 20]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 20, 10]}]
            }
            Job.check_data_quality(job_client_datasets)

        except:
            self.assertTrue(False, 'Raised an error while checking transposed images')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

