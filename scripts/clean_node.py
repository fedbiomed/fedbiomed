#!/usr/bin/env python


import os
import shutil

from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR

mp_spdz = os.path.join(ROOT_DIR, 'modules', 'MP-SPDZ')
for entry in [CONFIG_DIR, CACHE_DIR, TMP_DIR, VAR_DIR] + \
             [os.path.join(mp_spdz, f) for f in os.listdir(mp_spdz)]:
    if os.path.isdir(entry):
        print("[INFO] Removing directory ", entry)
        shutil.rmtree(entry)
    elif os.path.lexists(entry):
        print("[INFO] Removing file ", entry)
        os.remove(entry)
    else:
        # never happens currently : import re-creates env directories
        print("[INFO] Directory does not exist ", entry)
