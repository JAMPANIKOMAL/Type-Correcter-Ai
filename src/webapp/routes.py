"""
Defines the web routes for the Flask application (the GUI).
"""

from flask import (
    Blueprint, render_template, request, jsonify
)
# Import the singleton corrector instance from our AI module
# This is the bridge between our web GUI and our AI model.
from src.autocorrect.predictor import corrector_instance

# Create a Blueprint named 'app'. This helps organize routes.
bp = Blueprint('app', __name__, url_prefix='/')

@bp.route('/', methods=['GET'])
def index():
    """
    Renders the main HTML page (the GUI).
    
    Returns:
        Rendered HTML template.
    """
    # This tells Flask to find 'index.html' in the 'templates' folder.
    return render_template('index.html')


@bp.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to handle correction requests.
    Expects JSON data: {"text": "some noisy text"}
    Returns JSON data: {"correction": "some corrected text"}
    """
    try:
        # Get the JSON data sent from the JavaScript front-end
        data = request.get_json()
        
        if not data or 'text' not in data:
            print("ERROR: No text provided in request.")
            return jsonify({"error": "No text provided"}), 400
            
        input_text = data['text']
        
        # Use the global corrector instance to make a prediction
        corrected_text = corrector_instance.predict(input_text)
        
        # Return the correction as JSON
        return jsonify({"correction": corrected_text})
        
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Server error"}), 500