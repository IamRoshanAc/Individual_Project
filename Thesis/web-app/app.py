# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import numpy as np
# import pandas as pd
# import sklearn

# app = Flask(__name__)
# CORS(app)

# # Load the model
# with open('bestmodel.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Load the CSV data
# csv_data = pd.read_csv('players_all2.csv')  # Adjust the path to your CSV file

# def format_market_value(value):
#     if value >= 1e6:
#         return f"€{value / 1e6:.1f}M"
#     elif value >= 1e3:
#         return f"€{value / 1e3:.1f}K"
#     else:
#         return f"€{value:.1f}"

# def parse_value(value_str):
#     """Convert value strings like '€1.2M' or '€800K' into numerical values."""
#     try:
#         if isinstance(value_str, str):
#             value_str = value_str.replace('€', '').replace('M', 'e6').replace('K', 'e3')
#             return float(eval(value_str))
#         else:
#             return float(value_str)
#     except:
#         return np.nan

# def find_closest_matches(predictions, csv_data, n=5):
#     # Convert 'Value' column to numerical format
#     csv_data['Value'] = csv_data['Value'].apply(parse_value)
    
#     # Compute the absolute differences between the predictions and the actual values
#     csv_data['Difference'] = csv_data['Value'].apply(lambda x: min([abs(x - pred) for pred in predictions]))
    
#     # Sort by the smallest difference and return the top n rows
#     closest_matches = csv_data.sort_values(by='Difference').head(n)
    
#     # Convert the closest matches to a list of dictionaries
#     closest_matches = closest_matches.to_dict(orient='records')
    
#     return closest_matches

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get data from the request
#         data = request.get_json()
        
#         # Ensure data is a list of dictionaries
#         if isinstance(data, list) and all(isinstance(item, dict) for item in data):
#             # Convert to DataFrame
#             input_data = pd.DataFrame(data)
            
#             # Check if all required features are present
#             required_features = ['Ball control', 'Dribbling / Reflexes', 'Total power', 'Shooting / Handling', 
#                                  'Age', 'Total mentality', 'Finishing', 'Passing / Kicking', 'Shot power', 
#                                  'International reputation']
#             if not all(feature in input_data.columns for feature in required_features):
#                 return jsonify({'error': 'Missing required features in input data.'}), 400
            
#             # Convert to numpy array
#             input_array = input_data[required_features].values
            
#             # Predict log market values
#             log_predictions = model.predict(input_array)
            
#             # Reverse the logarithmic transformation to get the original market values
#             predicted_market_values = np.exp(log_predictions) 
            
#             # Apply the formatting function to the predicted market values
#             formatted_predictions = [format_market_value(value) for value in predicted_market_values]
            
#             # Find the closest matches in the CSV data
#             closest_matches = find_closest_matches(predicted_market_values, csv_data)
            
#             return jsonify({
#                 'predictions': formatted_predictions,
#                 'closest_matches': closest_matches
#             })
#         else:
#             return jsonify({'error': 'Invalid input format. Expected a list of dictionaries.'}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

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
            predicted_market_values = np.exp(log_predictions) *9
            
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