import json
from mkdocs.config.defaults import get_schema
from mkdocs.config.base import Config

# Read MkDocs yml file to extract URL information
with open('mkdocs.yml', 'rb') as conf:
    cfg = Config(schema=get_schema(), config_file_path='../../mkdocs.yml')
    cfg.load_file(conf)

navigation = cfg.get('nav')

top_nav = [nav for nav in navigation if 'Top-Bar' in nav]
footer_nav = [nav for nav in navigation if 'Footer' in nav]

menu = {
    'top': top_nav[0],
    'footer': footer_nav[0],
}
print(json.dumps(menu))
