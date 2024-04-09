#!/usr/bin/env python


import os
import shutil

from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR

data_keep_files = ['.gitignore', 'README.md', 'pseudo_adni_mod.csv', 'create_node_data.py']
mp_spdz = os.path.join(ROOT_DIR, 'modules', 'MP-SPDZ')

for entry in [CONFIG_DIR, CACHE_DIR, TMP_DIR, VAR_DIR] + \
             [os.path.join(mp_spdz, f) for f in os.listdir(mp_spdz)] + \
             [f.path
              for f in os.scandir(os.path.join(ROOT_DIR, 'notebooks', 'data'))
              if f.is_file() and f.name not in data_keep_files ] + \
             [e.path
              for d in os.scandir(os.path.join(ROOT_DIR, 'notebooks', 'data')) if d.is_dir()
              for e in os.scandir(d) if e.name not in data_keep_files ]:
    if os.path.isdir(entry):
        print("[INFO] Removing directory ", entry)
        shutil.rmtree(entry)
    elif os.path.lexists(entry):
        print("[INFO] Removing file ", entry)
        os.remove(entry)
    else:
        # never happens currently : import re-creates env directories
        print("[INFO] Directory does not exist ", entry)
