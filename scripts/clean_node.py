#!/usr/bin/env python


import os
import shutil

from fedbiomed.node.environ import environ

for _key in 'CONFIG_DIR', 'CACHE_DIR', 'TMP_DIR', 'VAR_DIR':
    dir = environ[_key]
    if os.path.isdir(dir):
        print("[INFO] Removing directory ", dir)
        shutil.rmtree(dir)
    else:
        # never happens currently : import re-creates env directories
        print("[INFO] Directory does not exist ", dir)
