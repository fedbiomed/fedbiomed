import os

import requests
from typing import Dict, Any, Tuple
import time
import random

class Repository:

    def __init__(self, uploads_url, tmp_dir: str, cache_dir: str):
        self.uploads_url = uploads_url
        self.tmp_dir = tmp_dir
        self.cache_dir = cache_dir


    def upload_file(self, filename: str) -> Dict[str, Any]:
        """
        uploads a file to a HTTP file repository.
        
        Returns:
            res (Dict[str, Any]): the result of the request under JSON
            format.
        """
        files = {'file': open(filename, 'rb')}
        res = requests.post(self.uploads_url, files=files)
        return res.json()

    def download_file(self, url: str, filename: str) -> Tuple[int, str]:
        """
        downloads a file from a HTTP file repository
        """
        res = requests.get(url)
        filepath = os.path.join(self.tmp_dir, filename)
        open(filepath, 'wb').write(res.content)
        return res.status_code, filepath
