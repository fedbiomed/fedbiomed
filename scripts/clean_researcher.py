#!/usr/bin/env python


import os
import shutil

from fedbiomed.common.constants import TENSORBOARD_FOLDER_NAME
from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR

for dir in CONFIG_DIR, \
        CACHE_DIR, \
        TMP_DIR, \
        os.path.join(ROOT_DIR, TENSORBOARD_FOLDER_NAME), \
        VAR_DIR, \
        os.path.join(ROOT_DIR, 'modules', 'MP-SPDZ'):
    if os.path.isdir(dir):
        print("[INFO] Removing directory ", dir)
        shutil.rmtree(dir)
    else:
        # never happens currently : import re-creates env directories
        print("[INFO] Directory does not exist ", dir)
