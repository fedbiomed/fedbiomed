"""API-only WSGI entry point: fedbiomed_node_api.wsgi:app."""

from .application import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
