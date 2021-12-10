import os
import sys
from flask import Flask

# Create Flask Application
app = Flask(__name__)

# DB prefix 
db_prefix = os.getenv('DB_PREFIX', 'db_')

app.config['NODE_CONFIG_FILE'] = os.getenv('NODE_CONFIG_FILE',
                                           "config_node.ini")

# Configuration of Flask APP to able to access Fed-BioMed node information
app.config['NODE_FEDBIOMED_ROOT'] = os.getenv('FEDBIOMED_ROOT', '/fedbiomed')

app.config['NODE_CONFIG_FILE_PATH'] = \
    os.path.join(app.config["NODE_FEDBIOMED_ROOT"],
                 'etc',
                 app.config['NODE_CONFIG_FILE'])

# Append fedbiomed root dir as a python path
sys.path.append(app.config['NODE_FEDBIOMED_ROOT'])

# Set config file path to mkae fedbiomed.common.environ to parse
# correct config file
os.environ["CONFIG_FILE"] = app.config['NODE_CONFIG_FILE_PATH']
from fedbiomed.node.environ import environ

# Set node information
app.config['NODE_ID'] = environ['NODE_ID']
app.config['NODE_DB_PATH'] = environ['DB_PATH']

# Data path where datafiles are stored.
app.config['DATA_PATH'] = \
    os.getenv('DATA_PATH',
              os.path.join(app.config['NODE_FEDBIOMED_ROOT'],
                           'data'))

app.config['DEBUG'] = os.getenv('DEBUG', 'True').lower() in \
                      ('true', 1, True, 'yes')
app.config['PORT'] = os.getenv('PORT', 8484)
app.config['HOST'] = os.getenv('HOST', '0.0.0.0')

# Log information for setting up a node connection
app.logger.info(f'Fedbiomed Node root dir has been set as \
        {app.config["NODE_FEDBIOMED_ROOT"]}')
app.logger.info(f'Fedbiomed Node config file is \
        {app.config["NODE_CONFIG_FILE"]}')
app.logger.info(f'Services are going to be configured for the node \
        {app.config["NODE_ID"]}')

# Import api route blueprint before importing routes
# and register as blueprint
from routes import api
app.register_blueprint(api)

# Run the applicaiton
if __name__ == '__main__':

    # Start Flask
    app.run(host=app.config['HOST'],
            port=app.config['PORT'])