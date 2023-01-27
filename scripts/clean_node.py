#!/usr/bin/env python


import os
import shutil

from fedbiomed.node.environ import environ

for dir in environ['CONFIG_DIR'], \
        environ['CACHE_DIR'], \
        environ['TMP_DIR'], \
        environ['VAR_DIR'], \
        os.path.join('ROOT_DIR', 'modules', 'MP-SPDZ', 'Player-Data'):
    if os.path.isdir(dir):
        print("[INFO] Removing directory ", dir)
        shutil.rmtree(dir)
    else:
        # never happens currently : import re-creates env directories
        print("[INFO] Directory does not exist ", dir)
