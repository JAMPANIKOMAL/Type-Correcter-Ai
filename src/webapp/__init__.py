"""
Web application package initializer (App Factory).

This file contains the 'create_app' factory function which
initializes and configures the Flask application.
"""

from flask import Flask
import os

def create_app():
    """
    Create and configure an instance of the Flask application.
    """
    # __name__ is the name of the current Python module
    app = Flask(__name__, instance_relative_config=True)
    
    # Set a default secret key (good for sessions)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # --- Register Blueprints (routes) ---
    # A blueprint is a way to organize a group of related
    # views and other code.
    from . import routes
    app.register_blueprint(routes.bp)

    print("Flask app created successfully.")
    return app