# Flask API Documentation

This Flask API provides a `/predict` endpoint for making predictions using the trained California Housing KNN model.

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r flask_requirements.txt
```

### 2. Ensure Model File Exists
The Flask app expects the model file `california_knn_pipeline.pkl` to be in the same directory as `flask_app.py`.

### 3. Run the Flask App
```bash
python flask_app.py
```

## API Endpoints

### POST /predict
Makes predictions on house prices.

**Request:**
```json
{
  "MedInc": 3.5,
  "HouseAge": 28.0,
  "AveRooms": 5.4,
  "AveBedrms": 1.1,
  "Population": 1425.5,
  "AveOccup": 3.07,
  "Latitude": 35.63,
  "Longitude": -119.57
}
```

**Response (Success):**
```json
{
  "prediction": 2.5234,
  "status": "success"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Usage Examples

### Using cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 3.5, "HouseAge": 28.0, "AveRooms": 5.4, "AveBedrms": 1.1, "Population": 1425.5, "AveOccup": 3.07, "Latitude": 35.63, "Longitude": -119.57}'
```

### Using Python
```python
import requests

data = {
    "MedInc": 3.5, "HouseAge": 28.0, "AveRooms": 5.4, 
    "AveBedrms": 1.1, "Population": 1425.5, "AveOccup": 3.07, 
    "Latitude": 35.63, "Longitude": -119.57
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

## Model Information

- **Type:** K-Nearest Neighbors Regressor
- **Parameters:** n_neighbors=9, weights=distance, p=1
- **Features:** 8 numeric values (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Output:** Predicted house price
