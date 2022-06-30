from datetime import timedelta
import os
from flask import Flask, render_template, send_from_directory
from flask_jwt_extended import JWTManager
from config import Config


build_dir = os.getenv('BUILD_DIR', '../ui/gui-build')

# Create Flask Application
app = Flask(__name__, static_folder=build_dir)

# Configure Flask app
db_prefix = os.getenv('DB_PREFIX', 'db_')
config = Config()
app.config.update(config.generate_config())

# Configure application to store JWTs in cookies. Whenever you make
# a request to a protected endpoint, you will need to send in the
# access or refresh JWT via a cookie or a header.
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']

app.config['JWT_COOKIE_SECURE'] = True

# Set the cookie paths, so that you are only sending your access token
# cookie to the access endpoints, and only sending your refresh token
# to the refresh endpoint.
app.config['JWT_ACCESS_COOKIE_PATH'] = '/api/'
app.config['JWT_REFRESH_COOKIE_PATH'] = '/token/refresh'
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

# Disable CSRF protection for this example. In almost every case,
# this is a bad idea. See examples/csrf_protection_with_cookies.py
# for how safely store JWTs in cookies
app.config['JWT_COOKIE_CSRF_PROTECT'] = True
# encryption relies on secret keys
# TODO: change it to use environment variable
app.config['SECRET_KEY'] = 'CHANGE ME'

jwt = JWTManager(app)


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
    # Start Flask
    app.run(host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG'])
