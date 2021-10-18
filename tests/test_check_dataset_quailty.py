from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet
import unittest

from unittest.mock import MagicMock, patch


class TestJobDatasetQualityCheck(unittest.TestCase):
    """
    Test `check_dataset_qualty` method of Job class
    Args:
        unittest ([type]): [description]
    """
    
    # Setup patchers for mocking modules
    def setUp(self):

        self.patcher1 = patch('fedbiomed.common.repository.Repository.upload_file')
        self.patcher2 = patch('fedbiomed.researcher.job.Job.validate_minimal_arguments')
        self.patcher3 = patch('fedbiomed.common.messaging.Messaging.__init__')
        self.patcher4 = patch('fedbiomed.common.messaging.Messaging.start')
        self.patcher5 = patch('fedbiomed.researcher.requests.Requests.__init__')
        
        self.mock_repository = self.patcher1.start()
        self.mock_job_validate = self.patcher2.start()
        self.mock_message_init = self.patcher3.start()
        self.mock_message_start = self.patcher4.start()
        self.mock_req_init = self.patcher5.start()

        pass

    # after the tests
    def tearDown(self):

        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        
        pass
    
    def test_check_quality_csv(self):
        
        """Test when federated csv datasets do match for federated training"""

        self.mock_message_init.return_value = None
        self.mock_message_start.return_value = None
        self.mock_req_init.return_value = None


        self.mock_repository.return_value.upload_file.return_value = {"file" : 'sss'}
        self.mock_job_validate.return_value = None

        model_file = MagicMock(return_value = None)
        model_file.save = MagicMock(return_value = '123')

        ## Test when every thing is okay
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}]
            }
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error when datasets have good quality')

         # Check when there are more than 2 datasets
        try:
            job_client_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}]
            } 
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)
        
        except:
             self.assertTrue(False, 'Raised an error when there are more than 3 \
                  datasets even they have good quality')


        # When csv datasets has diffrent number of rows
        try:
            job_client_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [100,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [15,5]}]
            }  
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)
        
        except:
             self.assertTrue(False, 'Raised an error when datasets have different number \
                                        of rows even they have good quality')


        # Dimension error when the dimensions are different
        with self.assertRaises(Exception):
            job_client_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,16]}]
            }

            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)
                
        # Check when the variable datatypes are differents
        with self.assertRaises(Exception):
            job_client_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'int' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'string'], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,6]}]
            }

            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        # Check when the datatypes are different
        with self.assertRaises(Exception):
            job_client_datasets = {
                'client-1' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,5]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [10,5]}],
                'client-3' : [{'data_type' : 'csv' , 'dtypes' : ['float', 'float' , 'float'], 'shape' : [10,6]}]
            }

            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)
    
    def test_check_quality_image(self):
        
        """Test when federated image dataset do match with good qualty"""

        self.mock_message_init.return_value = None
        self.mock_message_start.return_value = None
        self.mock_req_init.return_value = None


        self.mock_repository.return_value.upload_file.return_value = {"file" : 'sss'}
        self.mock_job_validate.return_value = None

        model_file = MagicMock(return_value = None)
        model_file.save = MagicMock(return_value = '123')


        ## Test when everything is okay
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
            }
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error when datasets have good quality')


        ## Test when there are more than 2 datasets 
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
            }
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error when datasets have good quality')
    

        ## Test when the number of samples are different
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,3, 10, 10]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 10, 10]}]
            }
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error when datasets have different sample sizes')
    

        # Check when color channels are different
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,1, 10, 10]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 10, 10]}]
            }

            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error while checking color channels')

        # Check when image shapes are different
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 10, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,3, 10, 25]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 10, 5]}]
            }
            
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error while checking image shapes')
        
        # Check when the images gets transposed
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 20, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,3, 10, 20]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 20, 10]}]
            }
            
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error while checking transposed images')


        # Check when image color channels and shapes are transposed
        try: 
            job_client_datasets = {
                'client-1' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1000,3, 20, 10]}],
                'client-2' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [1500,1, 10, 20]}],
                'client-3' : [{'data_type' : 'images' , 'dtypes' : [], 'shape' : [2000,3, 20, 10]}]
            }
            
            fds = FederatedDataSet(job_client_datasets)
            Job(data=fds, model = model_file)

        except:
            self.assertTrue(False, 'Raised an error while checking transposed images')



if __name__ == '__main__':  # pragma: no cover
    unittest.main()

