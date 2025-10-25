"""
Main entry point for the Type Correcter Ai web application.

This script imports the Flask app instance from the 'src.webapp' package
and runs the development server.

To run the application:
1. Activate your Python environment (e.g., `conda activate ...` or `source venv/bin/activate`)
2. Run this file: `python app.py`
"""

from src.webapp import create_app

# Create the Flask app instance using the app factory pattern
app = create_app()

if __name__ == "__main__":
    # Run the Flask app
    # host='0.0.0.0' makes it accessible on the local network.
    # debug=True provides auto-reloading and detailed error pages.
    app.run(host='0.0.0.0', port=5000, debug=True)