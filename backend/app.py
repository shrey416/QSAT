# -*- coding: utf-8 -*-
"""
app.py: Flask application for Soil Spectrometer Analysis.
Provides API endpoints for prediction, metrics, and feature rankings.
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import mymodel_utils # Import the utility functions

app = Flask(__name__)
# Allow requests from your frontend domain in production
# For development, allow from localhost:3000 (adjust port if needed)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:5173"]}}) # Replace with your frontend URL

# --- Application Initialization ---
# Run initialization when the app starts.
# In production with multiple workers (Gunicorn), this might run per worker.
# The lock inside initialize_application handles concurrency.
# For Vercel, this runs when the serverless function initializes.
initialization_successful = mymodel_utils.initialize_application()

if not initialization_successful:
    print("WARNING: Application failed to initialize properly. Endpoints might not work.")

# --- API Routes ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    initialized = mymodel_utils.get_status()
    if initialized:
        return jsonify({"status": "OK", "message": "Application initialized"}), 200
    else:
        return jsonify({"status": "Error", "message": "Application failed to initialize"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_soil():
    """
    Endpoint to receive spectral data and water level, return predictions.
    Expects JSON: { "waterLevel": int, "wavelengths": {"410": float, "535": float, ...} }
    """
    if not mymodel_utils.get_status():
        return jsonify({"error": "Service not ready, initialization failed."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON body found."}), 400

        water_level = data.get('waterLevel')
        wavelength_data = data.get('wavelengths')

        # --- Input Validation ---
        if water_level is None or not isinstance(water_level, (int, float)):
             return jsonify({"error": "Invalid request: 'waterLevel' missing or not a number."}), 400
        # Convert to int if possible (model expects specific levels)
        try:
             water_level = int(water_level)
             if water_level not in mymodel_utils.WATER_LEVELS_TO_PROCESS:
                  return jsonify({"error": f"Invalid request: 'waterLevel' must be one of {mymodel_utils.WATER_LEVELS_TO_PROCESS}."}), 400
        except ValueError:
             return jsonify({"error": "Invalid request: 'waterLevel' could not be converted to an integer."}), 400


        if not wavelength_data or not isinstance(wavelength_data, dict):
            return jsonify({"error": "Invalid request: 'wavelengths' missing or not a dictionary."}), 400

        # Validate wavelength keys and values
        valid_spectral_keys = set(mymodel_utils.SPECTRAL_COLS)
        provided_keys = set(wavelength_data.keys())
        invalid_keys = provided_keys - valid_spectral_keys
        if invalid_keys:
            return jsonify({"error": f"Invalid spectral keys provided: {list(invalid_keys)}. Valid keys are: {mymodel_utils.SPECTRAL_COLS}"}), 400

        # Check number of inputs (frontend should also validate this)
        num_provided = len(provided_keys)
        MIN_INPUTS = 2 # Model requirement
        MAX_INPUTS = len(mymodel_utils.SPECTRAL_COLS)
        if not (MIN_INPUTS <= num_provided <= MAX_INPUTS):
            return jsonify({"error": f"Must provide between {MIN_INPUTS} and {MAX_INPUTS} spectral values. Provided: {num_provided}"}), 400

        # Convert values to float and check for non-numeric inputs
        processed_wavelengths = {}
        for key, value in wavelength_data.items():
            try:
                processed_wavelengths[key] = float(value)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numeric value for wavelength '{key}': {value}"}), 400

        # --- Run Prediction ---
        status_info, predictions = mymodel_utils.run_prediction(processed_wavelengths, water_level)

        # --- Format Response ---
        # Combine status info with actual predictions
        response_data = {**status_info, **predictions}

        # Map internal target names to frontend keys if they differ
        # Example mapping (adjust based on your frontend's `METRIC_PARAM_KEYS`)
        frontend_key_map = {
            'Ph': 'pH',
            'Nitro': 'nitro',
            'Posh Nitro': 'phosphorus', # Assuming 'Posh Nitro' means Phosphorus
            'Pota Nitro': 'potassium', # Assuming 'Pota Nitro' means Potassium
            'Capacitity Moist': 'capacityMoist',
            'Temp': 'temperature',
            'Moist': 'moisture',
            'EC': 'electricalConductivity'
        }

        formatted_response = {
             'Prediction_Status': response_data.get('Prediction_Status', 'Unknown Error'),
             'Input_Water_Level': response_data.get('Input_Water_Level', water_level),
             'Provided_Features': response_data.get('Provided_Features', list(processed_wavelengths.keys())),
             'Imputed_Features': response_data.get('Imputed_Features', [])
        }
        for model_key, frontend_key in frontend_key_map.items():
             pred_value = response_data.get(model_key)
             # Return null for NaN or None (JSON standard)
             formatted_response[frontend_key] = None if (pred_value is None or (isinstance(pred_value, float) and np.isnan(pred_value))) else pred_value


        if "Error" in formatted_response['Prediction_Status'] or "Failed" in formatted_response['Prediction_Status']:
             # Return a more indicative HTTP status code for errors during prediction
             return jsonify(formatted_response), 500
        elif "Partial Success" in formatted_response['Prediction_Status']:
             # Could return 207 Multi-Status, but 200 is also acceptable
              return jsonify(formatted_response), 200
        else:
             return jsonify(formatted_response), 200

    except Exception as e:
        print(f"ERROR in /api/analyze: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns the pre-calculated model performance metrics."""
    if not mymodel_utils.get_status():
        return jsonify({"error": "Service not ready, initialization failed."}), 503

    metrics = mymodel_utils.get_performance_metrics()
    if "error" in metrics:
        return jsonify(metrics), 500 # If metrics loading failed during init

    # No reformatting needed if mymodel_utils saves in the correct structure
    return jsonify(metrics), 200


@app.route('/api/top-wavelengths', methods=['GET'])
def get_top_wavelengths():
    """
    Returns the top N ranked wavelengths for a given attribute.
    Query Params: attribute (e.g., 'pH', 'nitro'), count (e.g., 5)
    """
    if not mymodel_utils.get_status():
        return jsonify({"error": "Service not ready, initialization failed."}), 503

    attribute_key_frontend = request.args.get('attribute') # e.g., 'pH', 'nitro'
    count_str = request.args.get('count')

    if not attribute_key_frontend:
        return jsonify({"error": "Missing 'attribute' query parameter."}), 400
    if not count_str:
        return jsonify({"error": "Missing 'count' query parameter."}), 400

    try:
        count = int(count_str)
        if count < 1: raise ValueError("Count must be positive.")
    except ValueError:
        return jsonify({"error": "'count' must be a positive integer."}), 400

    # --- Map Frontend Attribute Key to Model Target Column Name ---
    # Inverse of the map used in /analyze
    model_target_map = {
        'pH': 'Ph',
        'nitro': 'Nitro',
        'phosphorus': 'Posh Nitro',
        'potassium': 'Pota Nitro',
        'capacityMoist': 'Capacitity Moist',
        'temperature': 'Temp',
        'moisture': 'Moist',
        'electricalConductivity': 'EC'
    }
    model_target_col = model_target_map.get(attribute_key_frontend)

    if not model_target_col:
        valid_frontend_keys = list(model_target_map.keys())
        return jsonify({"error": f"Invalid 'attribute' provided: {attribute_key_frontend}. Valid attributes: {valid_frontend_keys}"}), 400


    rankings = mymodel_utils.get_feature_rankings()
    if "error" in rankings:
        return jsonify(rankings), 500 # If ranking loading failed

    target_rankings = rankings.get(model_target_col)

    if target_rankings is None:
         # This might happen if ranking failed for this specific target during init
         return jsonify({"error": f"Ranking data not available for attribute '{attribute_key_frontend}' (target: {model_target_col})."}), 404
    elif not isinstance(target_rankings, list):
         # Data integrity check
         print(f"Warning: Unexpected format for rankings of {model_target_col}. Expected list, got {type(target_rankings)}")
         return jsonify({"error": f"Internal server error retrieving rankings for '{attribute_key_frontend}'."}), 500

    # Slice the rankings to get the top N
    top_rankings = target_rankings[:count]

    return jsonify(top_rankings), 200

@app.route('/api/metrics', methods=['GET'])
def get_metrics_v2():
    """Returns the pre-calculated model performance metrics."""
    if not mymodel_utils.get_status():
        return jsonify({"error": "Service not ready, initialization failed."}), 503

    metrics = mymodel_utils.get_performance_metrics()
    if "error" in metrics:
        return jsonify(metrics), 500

    # --- Add Debug Print ---
    print("--- Serving /api/metrics ---")
    print(f"Type of metrics object: {type(metrics)}")
    # Log a sample to see if it's correct before jsonify
    try:
         print(f"Sample metric (R2, 0ml, Ph): {metrics.get('R2', {}).get('0ml', {}).get('Ph')}")
    except Exception as e:
         print(f"Error accessing sample metric: {e}")
    print("--- End Serving /api/metrics ---")
    # --- End Debug Print ---

    return jsonify(metrics), 200

# --- START OF NEW CODE FOR GEMINI INTEGRATION ---

# Load environment variables (for API key)
from dotenv import load_dotenv
load_dotenv() # Load variables from .env file

import google.generativeai as genai

# --- Gemini Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_configured = False
gemini_model = None

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables. /api/get-insights endpoint will not work.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Initialize the model you want to use
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or another suitable model
        gemini_configured = True
        print("Gemini AI configured successfully.")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini AI: {e}")
        gemini_configured = False

@app.route('/api/get-insights', methods=['POST'])
def get_gemini_insights():
    """
    Endpoint to receive a prompt (initial soil data or user message)
    and get insights from Gemini.
    Expects JSON: { "message": string }
    """
    if not mymodel_utils.get_status():
        return jsonify({"error": "Soil analysis service not ready."}), 503

    if not gemini_configured or gemini_model is None:
        return jsonify({"error": "Gemini AI service is not configured or available."}), 503

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request: 'message' missing in JSON body."}), 400

        user_message = data['message']
        if not isinstance(user_message, str) or not user_message.strip():
            return jsonify({"error": "Invalid request: 'message' must be a non-empty string."}), 400

        print(f"Sending to Gemini: {user_message[:100]}...") # Log truncated message

        # --- Call Gemini API ---
        # Consider adding more robust error handling for API calls
        # For conversational chat, you might manage history differently,
        # but for simple Q&A, generating based on the single message is fine.
        response = gemini_model.generate_content(user_message)

        # Basic check if the response has text
        # More complex checks might be needed depending on the Gemini model/response structure
        if response and hasattr(response, 'text') and response.text:
            print("Received response from Gemini.")
            return jsonify({"response": response.text}), 200
        else:
             # Log the full response if it's unusual
            print(f"Warning: Received unexpected or empty response from Gemini: {response}")
            return jsonify({"error": "Received no content from Gemini AI."}), 500

    except Exception as e:
        print(f"ERROR in /api/get-insights: {e}")
        import traceback
        traceback.print_exc()
        # Be careful not to expose sensitive details in production error messages
        return jsonify({"error": "An error occurred while communicating with the Gemini AI service."}), 500

# Optionally, update health check to include Gemini status
@app.route('/api/health/v2', methods=['GET']) # New route to avoid breaking old one
def health_check_v2():
    """Basic health check endpoint including Gemini status."""
    soil_initialized = mymodel_utils.get_status()
    status_code = 200
    response_data = {
        "soil_service_status": "OK" if soil_initialized else "Error",
        "gemini_service_status": "OK" if gemini_configured else "Error",
        "message": []
    }
    if soil_initialized:
        response_data["message"].append("Soil analysis application initialized.")
    else:
        response_data["message"].append("Soil analysis application failed to initialize.")
        status_code = 500
    if gemini_configured:
         response_data["message"].append("Gemini AI service configured.")
    else:
         response_data["message"].append("Gemini AI service NOT configured (check API key).")
         # Don't necessarily make the whole health check fail if Gemini is down
         # status_code = 500 # Uncomment if Gemini is critical

    return jsonify(response_data), status_code


# --- END OF NEW CODE FOR GEMINI INTEGRATION ---

# Make sure the following lines are the VERY LAST lines in the file
if __name__ == '__main__':
    # Use a production-ready server like Gunicorn or Waitress instead of app.run()
    # For local development:
    app.run(debug=True, port=5000) # Set debug=False in production