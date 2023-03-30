# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""HTTP file repository from which to upload and download files."""

import os
import requests  # Python built-in library

from json import JSONDecodeError
from typing import Callable, Dict, Any, Tuple, Text, Union, Optional

from fedbiomed.common.exceptions import FedbiomedRepositoryError
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.logger import logger


class Repository:
    """HTTP file repository from which to upload and download files.

    Files are uploaded from/downloaded to a temporary file (`tmp_dir`).
    Data uploaded should be:

    - python code (*.py file) that describes model +
        data handling/preprocessing
    - model params (under *.pt format)
    """
    def __init__(self,
                 uploads_url: Union[Text, bytes],
                 tmp_dir: str,
                 cache_dir: str):
        """Constructor of the class.

        Args:
            uploads_url: The URL where we upload files
            tmp_dir: A directory for temporary files
            cache_dir: Currently unused
        """

        self.uploads_url = uploads_url
        self.tmp_dir = tmp_dir
        self.cache_dir = cache_dir  # unused

    def upload_file(self, filename: str) -> Dict[str, Any]:
        """Uploads a file to an HTTP file repository (through an HTTP POST request).

        Args:
            filename: A name/path of the file to upload.

        Returns:
            The result of the request under JSON format.

        Raises:
            FedbiomedRepositoryError: unable to read the file 'filename'
            FedbiomedRepositoryError: POST HTTP request fails or returns
                an HTTP status 4xx (bad request) or 500 (internal server error)
            FedbiomedRepositoryError: unable to deserialize JSON from
                the request
        """
        # first, we are trying to open the file `filename` and catch
        # any known exceptions related top `open` builtin function
        try:
            files = {'file': open(filename, 'rb')}
        except FileNotFoundError:
            _msg = ErrorNumbers.FB604.value + f': File {filename} not found, cannot upload it'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except PermissionError:
            _msg = ErrorNumbers.FB604.value + f': Unable to read {filename} due to unsatisfactory privileges' + \
                ", cannot upload it"
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except OSError:
            _msg = ErrorNumbers.FB604.value + f': Cannot read file {filename} when uploading'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)

        # second, we are issuing an HTTP 'POST' request to the HTTP server

        _res = self._request_handler(requests.post, self.uploads_url,
                                     filename, files=files)
        # checking status of HTTP request

        self._raise_for_status_handler(_res, filename)

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
        """Downloads a file from a HTTP file repository (through an HTTP GET request).

        Args:
            url: An url from which to download file
            filename: The name of the temporary file

        Returns:
            status: The HTTP status code
            filepath: The complete pathfile under which the temporary file is saved
        """

        res = self._request_handler(requests.get, url, filename)
        self._raise_for_status_handler(res, filename)
        filepath = os.path.join(self.tmp_dir, filename)

        try:
            open(filepath, 'wb').write(res.content)
        except FileNotFoundError as err:
            _msg = ErrorNumbers.FB604.value + str(err) + ', cannot save the downloaded content into it'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except PermissionError:
            _msg = ErrorNumbers.FB604.value + f': Unable to read {filepath} due to unsatisfactory privileges'
            ", cannot write the downloaded content into it"
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except MemoryError:
            _msg = ErrorNumbers.FB604.value + f" : cannot write on {filepath}: out of memory!"
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except OSError:
            _msg = ErrorNumbers.FB604.value + f': Cannot open file {filepath} after downloading'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)

        return res.status_code, filepath

    def _raise_for_status_handler(self, response: requests, filename: str = ''):
        """Handler that deals with exceptions.

        Also raises the appropriate
        exception if the HTTP request has failed with a code error (e.g. 4xx or 500)

        Args:
            response: The HTTP request's response (eg `requests.post` result).
            filename: The name of the file that is uploaded/downloaded,
                (regarding the HTTP request issued).
                Defaults to ''.

        Raises:
            FedbiomedRepositoryError: if request fails, raises an FedBioMedError
                with the appropriate code error/ message
        """
        _method_msg = Repository._get_method_request_msg(response.request.method)
        try:
            # `raise_for_status` method raises an HTTPError if the status code
            # is 4xx or 500
            response.raise_for_status()
        except requests.HTTPError as err:
            if response.status_code == 404:
                # handling case where status code of HTTP request equals 404
                _msg = ErrorNumbers.FB202.value + f' when {_method_msg} {filename}'

            else:
                # handling case where status code of HTTP request is 4xx or 500
                _msg = ErrorNumbers.FB203.value + f' when {_method_msg} {filename}' +\
                    f'(status code: {response.status_code})'
            logger.error(_msg)
            logger.debug('Details of exception: ' + str(err))
            raise FedbiomedRepositoryError(_msg)
        else:
            action = {"GET": "download", "PUT": "upload"}.get(
                response.request.method, f"HTTP {response.request.method} request"
            )
            logger.debug(f"{action} of file {filename} successful, with status code {response.status_code}")

    @staticmethod
    def _get_method_request_msg(req_type: str) -> str:
        """Returns the appropriate message whether the HTTP request is GET (downloading) or POST (uploading).

        Args:
            req_type: The request type ('GET', 'POST')

        Returns:
            The appropriate message (that will be used for the error message description if any error has been found)
        """
        # FIXME: this method only provide messages for the HTTP request 'POST' and
        # 'GET'. It should be completed as long other methods based on other requests
        # are added in the class (eg 'PUT' or 'DELETE' HTTP requests)
        if req_type.upper() == "POST":
            method_msg = "uploading file"
        elif req_type.upper() == "GET":
            method_msg = "downloading file"
        else:
            method_msg = 'issuing unknown HTTP request'
        return method_msg

    def _request_handler(self,
                         http_request: Callable,
                         url: str,
                         filename: str,
                         *args: Optional[Any],
                         **kwargs: Optional[Any]) -> requests:
        """Handles error that can trigger if the HTTP request fails (e.g. if request exceeded timeout, ...).

        Args:
            http_request: The requests HTTP method (callable)
            url: The url method to which to connect to
            filename: The name of the file to upload / download
            *args: The positional arguments to be passed to the callable method.
            **kwargs: The named arguments to be passed to the callable method.

        Returns:
           The result of the request if request is successful

        Raises:
            FedbiomedRepositoryError: - Timeout exceeded.
                - Too many redirects.
                - URL is badly written, or missing some parts (eg: missing scheme).
                - The connection is unsuccessful, when the service to connect is unknown.
                - Catches other exceptions coming from requests package
        """
        req_method = getattr(http_request, '__name__')
        req_method = req_method.upper()
        _method_msg = Repository._get_method_request_msg(req_method)

        try:
            # issuing the HTTP request
            res = http_request(url, *args, **kwargs)
        except requests.Timeout:
            # request exceeded timeout set
            _msg = ErrorNumbers.FB201.value + f' : {req_method} HTTP request time exceeds Timeout'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except requests.TooManyRedirects:
            # request had too many redirections
            _msg = ErrorNumbers.FB201.value + f' : {req_method} HTTP request exceeds max number of redirection'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except (requests.URLRequired, ValueError) as err:
            # request has been badly formatted
            _msg = ErrorNumbers.FB604.value + f" : bad URL when {_method_msg} {filename}" + \
                "(details :" + str(err) + " )"
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        except requests.ConnectionError:
            # an error during connection has occurred
            _msg = ErrorNumbers.FB201.value + f' when {_method_msg} {filename}' + \
                f' to {self.uploads_url}: name or service not known'
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)

        except requests.RequestException as err:
            # requests.ConnectionError should catch all exceptions
            # triggered by `requests` package
            _msg = ErrorNumbers.FB200.value + f': when {_method_msg} {filename}' + \
                f' (HTTP {req_method} request failed). Details: ' + str(err)
            logger.error(_msg)
            raise FedbiomedRepositoryError(_msg)
        return res
