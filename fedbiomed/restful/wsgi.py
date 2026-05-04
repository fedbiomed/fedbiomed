from .application import app
from .config import config

# Production
if __name__ == "__main__":
    from .config import config

    app.run(debug=config["DEBUG"])
