from .application import app

# Production
if __name__ == "__main__":
    from .config import config

    app.run(debug=config["DEBUG"])
