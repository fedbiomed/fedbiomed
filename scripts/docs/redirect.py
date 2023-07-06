#!/usr/bin/env python

"""

This module recrate redirect.htm files the URLs that does not include 
any `latest` or `version` base URI. 
"""


import argparse
import os
import glob
from jinja2 import Template

parser = argparse.ArgumentParser()


parser.add_argument('-src', '--source') 
parser.add_argument('-base', '--base')
parser.add_argument('-buri', '--base-uri')

args=parser.parse_args()


def produce_redirection(base, uri, path):

     with open('scripts/docs/redirect.html', 'rb') as f:
        template= Template(f.read().decode('utf-8'), 
                           autoescape=True,
                           keep_trailing_newline=True)
        url=path.replace(base, '')

        if uri == "../":
            url_splited = "/".join(url.strip("/").split('/')[1:])
            reldst=os.path.relpath(f"/{url_splited}", url)
        else:
            reldst=os.path.relpath(f"{uri}{os.path.sep}{url}", url)

        # Relative path
        href = '/'.join(reldst.split(os.path.sep))

        index_html = template.render(href=href)
        
        # Overwrite index html
        with open(os.path.join(path, 'index.html'), '+w') as file:
            file.write(index_html)
            file.close()

        # Remove md files and ipynb files from save same space from disk
        files= glob.glob((os.path.join(path, '*.md')))
        for f in files:
            os.remove(f)

        files= glob.glob((os.path.join(path, '*.ipynb')))
        for f in files:
            os.remove(f)

if os.path.isfile(args.source):
    produce_redirection(args.base, args.base_uri, os.path.dirname(args.source))


for (dirpath, dirnames, filenames) in os.walk(args.source):
    produce_redirection(args.base, args.base_uri, dirpath)
   