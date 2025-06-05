from .config import config
from .application import app

# Production
if __name__ == '__main__':
    app.run(debug=config["DEBUG"])
