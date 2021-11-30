import os
import sys
from utils import get_node_id
from flask import Flask


# Create Flask Application
app = Flask(__name__)

# DB prefix 
db_prefix = 'db_'

# Configuration of Flask APP to able to access Fed-BioMed node information
app.config['NODE_FEDBIOMED_ROOT']     = os.getenv('FEDBIOMED_ROOT' , '/fedbiomed')
app.config['NODE_CONFIG_FILE']        = os.getenv('NODE_CONFIG_FILE', 'config_node.ini')
app.config['NODE_CONFIG_FILE_PATH']   = os.path.join(app.config['NODE_FEDBIOMED_ROOT'], 
                                                    'etc' , app.config['NODE_CONFIG_FILE'])

# Set config file path to mkae fedbiomed.common.environ to parse correct config file
os.environ["CONFIG_FILE"] = app.config['NODE_CONFIG_FILE_PATH']

# Data path where datafiles are stored. 
app.config['DATA_PATH']               = os.getenv('DATA_PATH' , 
                                                    os.path.join(
                                                            app.config['NODE_FEDBIOMED_ROOT'], 
                                                            'data') )

app.config['NODE_ID']                 = get_node_id(app.config['NODE_CONFIG_FILE_PATH'])
app.config['NODE_DB_PATH']            = os.path.join(app.config['NODE_FEDBIOMED_ROOT'], 'var', db_prefix + app.config['NODE_ID'] + '.json')
app.config['DEBUG']                   = os.getenv('DEBUG', 'True').lower() in ('true' , 1, True, 'yes')
app.config['PORT']                    = os.getenv('PORT', 8484)
app.config['HOST']                    = os.getenv('HOST', '0.0.0.0')

# Log information for setting up a node connection
app.logger.info(f'Fedbiomed Node root dir has been set as {app.config["NODE_FEDBIOMED_ROOT"]}')
app.logger.info(f'Fedbiomed Node config file is {app.config["NODE_CONFIG_FILE"]}')
app.logger.info(f'Services are going to be configured for the node {app.config["NODE_ID"]}')



# Append fedbiomed root dir as a python path
sys.path.append(app.config['NODE_FEDBIOMED_ROOT'])




# Import api route blueprint before importing routes 
# and register as blue prnt
from routes import api
app.register_blueprint(api)





if __name__ == '__main__':

    # Start Flask
    app.run(host = app.config['HOST'], 
            port = app.config['PORT'])  

   
   