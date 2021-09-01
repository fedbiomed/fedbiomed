import os

import requests  # Python built-in library
from typing import Dict, Any, Tuple, Text, Union

class Repository:
    """HTTP file repository from which to upload and download files
    """
    def __init__(self,
                 uploads_url: Union[Text, bytes],
                 tmp_dir: str,
                 cache_dir: str):
        
        self.uploads_url = uploads_url
        self.tmp_dir = tmp_dir
        self.cache_dir = cache_dir  #unused


    def upload_file(self, filename: str) -> Dict[str, Any]:
        """
        uploads a file to a HTTP file repository (through an
        HTTP POST request).
        
        Returns:
            res (Dict[str, Any]): the result of the request under JSON
            format.
        """
        files = {'file': open(filename, 'rb')}
        res = requests.post(self.uploads_url, files=files)
        return res.json()

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
