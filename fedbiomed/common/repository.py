import os

import requests  # Python built-in library
from typing import Dict, Any, Tuple, Text, Union
from fedbiomed.common.logger import logger

from fedbiomed.common.exceptions import FedbiomedRepositoryError
from fedbiomed.common.constants import ErrorNumbers
from json import JSONDecodeError


class Repository:
    """HTTP file repository from which to upload and download files.
    Files are uploaded from/dowloaded to a temporary file (`temp_fir`)
    Data uploaded should be:
    - python code (*.py file) that describes model +
    data handling/preprocessing
    - model params (under *.pt format)
    """
    def __init__(self,
                 uploads_url: Union[Text, bytes],
                 tmp_dir: str,
                 cache_dir: str):
        
        self.uploads_url = uploads_url
        self.tmp_dir = tmp_dir
        self.cache_dir = cache_dir  # unused

    def upload_file(self, filename: str) -> Dict[str, Any]:
        """
        uploads a file to a HTTP file repository (through an
        HTTP POST request).
        Args:
            filename (str): name/path of the file to upload.
        Returns:
            res (Dict[str, Any]): the result of the request under JSON
            format.
        Raises: 
            FedbiomedRepositoryError: when unable to read the file 'filename'
            FedbiomedRepositoryError: when POST HTTP request fails or returns
            a HTTP status 4xx (bad request) or 500 (internal server error)
            FedbiomedRepositoryError: when unable to deserialize JSON from
            the request
        """
        # first, we are trying to open the file `filename` and catch
        # any known exceptions related top `open` builtin function
        try:
            files = {'file': open(filename, 'rb')}
        except FileNotFoundError:
            _msg = ErrorNumbers.FB603.value + f': File {filename} not found, cannot upload it'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except PermissionError:
            _msg = ErrorNumbers.FB603.value + f': Unable to read {filename} due to unsatisfactory privileges'
            ", cannot upload it"
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except OSError:
            _msg = ErrorNumbers.FB603.value + f': Cannot read file {filename} when uploading'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        
        # second, we are issuing an HTTP 'POST' request to the HTTP server
        try:
            _res = requests.post(self.uploads_url, files=files)
        except requests.Timeout:
            # request exceeded timeout set 
            _msg = ErrorNumbers.FB201 + ' : requests time exceed Timeout'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except requests.TooManyRedirects:
            # request had too any redirections
            _msg = ErrorNumbers.FB201 + ' : requests time exceed number of redirection'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except requests.URLRequired:
            # request has been badly formatted
            _msg = ErrorNumbers.FB603.value + f" : URL not specified when uploading file {filename}"
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except requests.ConnectionError:
            # an error during connection has occured
            _msg = ErrorNumbers.FB201.value + f' when uploading file {filename},' 
            ' name or service not known'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        
        except requests.RequestException as err:
            # requests.ConnectionError should catch all exceptions
            # triggered by `requests` package
            _msg = ErrorNumbers.FB200.value + f': when uploading file {filename}'
            ' (HTTP POST request failed). Details: ' + str(err)
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        
        # checking status of HTTP request
        try:
            # `raise_for_status` method raises an HTTPError if the status code 
            # is 4xx or 500
            _res.raise_for_status()
        except requests.HTTPError as err:
            if _res.status_code == 404:
                # handling case where status code of HTTP request equals 404
                _msg = ErrorNumbers.FB202.value + f' when uploading file {filename}'
                
            else:
                # handling case where status code of HTTP request is 4xx or 500
                _msg = ErrorNumbers.FB203.value + f' when uploading file {filename}'
                f'(status code: {_res.status_code})'
            
            logger.error(_msg)
            logger.debug('Details of exception: ' + str(err))
            raise FedbiomedRepositoryError(_msg)

        else:
            logger.debug(f'upload (HTTP POST request) of file {filename} successful,' 
                         f' with status code {_res.status_code}')
            
        # finally, we are deserializing message from JSON
        try:
            json_res = _res.json()
        except JSONDecodeError:
            # might be triggered by `request` package when deserializing
            _msg = 'Unable to deserialize JSON from HTTP POST request (when uploading file)'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        return json_res

    def download_file(self, url: str, filename: str) -> Tuple[int, str]:
        """
        downloads a file from a HTTP file repository (
            through an HTTP GET request)
        
        Args:
            url (str): url from which to download file
            filename (str): name of the temporary file
            
        Returns:
            status (int): HTTP status code
            filepath (str): the complete pathfile under
            which the temporary file is saved
        """
        res = requests.get(url)
        filepath = os.path.join(self.tmp_dir, filename)
        open(filepath, 'wb').write(res.content)
        return res.status_code, filepath
