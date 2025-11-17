import flask
import pickle
import numpy as np
from flask import request, jsonify

# Initialize Flask app
app = flask.Flask(__name__)

# Load the pre-trained model
model = None

def load_model():
    global model
    try:
        with open('california_knn_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model file 'california_knn_pipeline.pkl' not found!")
        model = None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

# Load model when app starts
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict house prices based on input features."""
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'error': 'Invalid request: No JSON data provided',
                'status': 'error'
            }), 400
        
        # Define required features
        required_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                           'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        # Check if all required features are present
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'required_features': required_features,
                'status': 'error'
            }), 400
        
        # Extract features in the correct order
        features = np.array([[data[f] for f in required_features]])
        
        # Validate that all values are numeric
        try:
            features = features.astype(float)
        except ValueError:
            return jsonify({
                'error': 'Invalid data types: All features must be numeric',
                'status': 'error'
            }), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
