import os

import requests
import time
import random

class Repository:

    def __init__(self, uploads_url, tmp_dir, cache_dir):
        self.uploads_url = uploads_url
        self.tmp_dir = tmp_dir
        self.cache_dir = cache_dir


    def upload_file(self, filename):
        """
        upload a file to a HTTP file repository
        """
        wait_time = random.randint(1,5)
        time.sleep(wait_time)
        files = {'file': open(filename, 'rb')}
        res = requests.post(self.uploads_url, files=files)
        return res.json()

    def download_file(self, url, filename):
        """
        download a file from a HTTP file repository
        """
        wait_time = random.randint(1,5)
        time.sleep(wait_time)
        res = requests.get(url)
        filepath = os.path.join(self.tmp_dir, filename)
        open(filepath, 'wb').write(res.content)
        return res.status_code, filepath
