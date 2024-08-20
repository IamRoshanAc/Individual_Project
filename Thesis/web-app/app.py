from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask-cors
import pickle
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire Flask app

# Load the model
with open('bestmodel.pkl', 'rb') as file:
    model = pickle.load(file)

def format_market_value(value):
    if value >= 1e6:
        return f"€{value / 1e6:.1f}M"
    elif value >= 1e3:
        return f"€{value / 1e3:.1f}K"
    else:
        return f"€{value:.1f}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        
        # Debug print
        print("Received data:", data)
        
        # Ensure data is a list of dictionaries
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Convert to DataFrame
            input_data = pd.DataFrame(data)
            
            # Check if all required features are present
            required_features = ['Ball control', 'Dribbling / Reflexes', 'Total power', 'Shooting / Handling', 
                                 'Age', 'Total mentality', 'Finishing', 'Passing / Kicking', 'Shot power', 
                                 'International reputation']
            if not all(feature in input_data.columns for feature in required_features):
                return jsonify({'error': 'Missing required features in input data.'}), 400
            
            # Debug print
            print("Input DataFrame:", input_data)
            
            # Convert to numpy array
            input_array = input_data[required_features].values
            
            # Debug print
            print("Input array:", input_array)
            
            # Predict log market values
            log_predictions = model.predict(input_array)
            
            # Debug print
            print("Log predictions:", log_predictions)
            
            # Reverse the logarithmic transformation to get the original market values
            predicted_market_values = np.exp(log_predictions) 
            
            # Debug print
            print("Predicted market values:", predicted_market_values)
            
            # Apply the formatting function to the predicted market values
            formatted_predictions = [format_market_value(value) for value in predicted_market_values]
            
            return jsonify({'predictions': formatted_predictions})
        else:
            return jsonify({'error': 'Invalid input format. Expected a list of dictionaries.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
