import datetime
import argparse
import os
import xml.etree.cElementTree as ET
from mkdocs.config.defaults import get_schema
from mkdocs.config.base import Config

# Parse version argument
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--version', action='store', help='Version string e.g v2.1', type=str, nargs='?', const='',
                    metavar='version')

args = parser.parse_args()
version = args.version if args.version is not None else ''

# Read MkDocs yml file to extract URL information
with open('mkdocs.yml', 'rb') as conf:
    cfg = Config(schema=get_schema(), config_file_path='../mkdocs.yml')
    cfg.load_file(conf)

cfg.load_dict({})
navigation = cfg.get('nav')
site_url = cfg.get('site_url')

# Read previous XML file if it is existed. if not create new
# xml file
sitemap_urls = []
if os.path.exists('sitemap.xml'):
    tree = ET.parse('sitemap.xml')
    root = tree.getroot()
    for t in root:
        for u in t:
            if 'loc' in u.tag:
                sitemap_urls.append(u.text)
else:
    # Initialize sitemap XML
    root = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    tree = ET.ElementTree(root)


# Get URLS from MkDocs config
def return_urls(navs, urls):
    for nav in navs:
        if isinstance(nav, dict):
            key = list(nav.keys())[0]
            if key != 'Footer':
                if isinstance(nav[key], list):
                    return_urls(nav[key], urls)
                else:
                    urls.append(nav[key])
    return urls


# Create URLS and write to sitemap.xml
urls = []
return_urls(navigation, urls)

# Ignore duplicates
urls = list(set(urls))

# Modification date, the date this script is run
lastmod_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Configure URLS
for url in urls:
    # Change relative path to absolute path
    url = url.replace('./', '/')

    # Home page and index news or index doc pages
    if 'index' in url:
        url = url.replace('index', '')

    # Remove file extension
    url = url.replace('.md', '')
    url = url.replace('.ipynb', '')

    # Add version if the URL is not public
    if 'news/' in url or 'pages/' in url or '/#' in url or url == '/' or url == '':
        url = site_url + url
    else:
        url = site_url + '/' + version + url

    # Make sure path are in correct format
    url = url.replace('//', '/')

    # If url is not exist in previous sitemap
    if not url in sitemap_urls:
        doc = ET.SubElement(root, "url")
        ET.SubElement(doc, "loc").text = url
        ET.SubElement(doc, 'lastmod').text = lastmod_date
        ET.SubElement(doc, 'changefreq').text = 'daily'

# Write final XML file
ET.register_namespace("", "http://www.sitemaps.org/schemas/sitemap/0.9")
tree.write("sitemap.xml")
