import os
import sys
from flask import Flask, render_template, send_from_directory

build_dir = os.getenv('BUILD_DIR', '../ui/gui-build')

# Create Flask Application
app = Flask(__name__, static_folder=build_dir)

# DB prefix 
db_prefix = os.getenv('DB_PREFIX', 'db_')
docker_status = os.getenv('DOCKER', 'False').lower() in ('true', '1')

# Configuring data path both for RW and SAVE by
# considering docker status. Docker status means, Is gui
# runs in a docker container
if docker_status:
    # If application is launched in docker container
    # Fed-BioMed root will be /fedbiomed
    app.config['NODE_FEDBIOMED_ROOT'] = '/fedbiomed'

    # Data path where datafiles are stored.
    app.config['DATA_PATH_RW'] = '/data'
    # Data path for saving dataset
    app.config['DATA_PATH_SAVE'] = os.getenv('DATA_PATH', '/data')

else:
    # Configuration of Flask APP to able to access Fed-BioMed node information
    app.config['NODE_FEDBIOMED_ROOT'] = os.getenv('FEDBIOMED_ROOT', '/fedbiomed')

    data_path = os.getenv('DATA_PATH')
    if not data_path:
        data_path = '/data'
    else:
        if data_path.startswith('/'):
            assert os.path.isdir(data_path), f'Given absolute "{data_path}" does not exist or it is not a directory.'
        else:
            data_path = os.path.join(app.config['NODE_FEDBIOMED_ROOT'], data_path)
            assert os.path.isdir(data_path), f'{data_path} has not been found in Fed-BioMed root directory or ' \
                                             f'it is not a directory. Please make sure that the folder is exist.'

    # Data path where datafiles are stored. Since node and gui
    # works in same machine without docker, path for writing and reading
    # will be same for saving into database
    app.config['DATA_PATH_RW'] = data_path
    app.config['DATA_PATH_SAVE'] = data_path
    print(os.getenv('DATA_PATH'))
    print(data_path)
# Get name of the config file defaul is "config_node.ini"
app.config['NODE_CONFIG_FILE'] = os.getenv('NODE_CONFIG_FILE',
                                           "config_node.ini")

# Exact configuration file path
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


@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>')
def index(path):
    """ The index route. This route should render the
        react build files. Which is located at the front-end folder.
    """
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

    return render_template('index.html')


@app.route('/static', methods=['GET'])
def static_react():
    """ Static route is the route for static files of the
        react front-end applicaltion
    """

    return 'HELLO STATIC FILES'


# Run the applicaiton
if __name__ == '__main__':
    # Start Flask
    app.run(host=app.config['HOST'],
            port=app.config['PORT'])
