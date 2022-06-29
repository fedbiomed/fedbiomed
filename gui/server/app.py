import os
from flask import Flask, render_template, send_from_directory
from config import Config
from flask_login import LoginManager
# from user import login


build_dir = os.getenv('BUILD_DIR', '../ui/gui-build')

# Create Flask Application
app = Flask(__name__, static_folder=build_dir)

# Configure Flask app
db_prefix = os.getenv('DB_PREFIX', 'db_')
config = Config()
app.config.update(config.generate_config())

# encryption relies on secret keys
# TODO: change it to use environment variable
app.config['SECRET_KEY'] = 'CHANGE ME'


# Import api route blueprint before importing routes
# and register as blueprint
from routes import api
app.register_blueprint(api)


# Routes for react build directory
@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>')
def index(path):
    """ The index route. This route should render the
        React build files. Which is located at the front-end folder.
    """
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

    return render_template('index.html')


# Run the application
if __name__ == '__main__':
    # Link the login instance to our app
    # login_manager.init_app(app)
    # login.login_view = 'login'
    # Start Flask
    app.run(host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG'])
