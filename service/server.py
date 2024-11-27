from flask import Flask, request, jsonify
import numpy as np
from joblib import load  # Assuming your `loaded_model` is stored as a pickle file

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('NSL-KDD/linear_svc_model.pkl', 'rb') as f:  # Replace 'model.pkl' with your model file path
    loaded_model = load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        if not data or 'x_test_point' not in data:
            return jsonify({'error': 'Invalid input, expected JSON with key "x_test_point"'}), 400

        # Convert input to numpy array
        x_test_point = np.array(data['x_test_point'])

        # Ensure it's 2D
        if x_test_point.ndim == 1:
            x_test_point = x_test_point.reshape(1, -1)

        # Perform prediction
        predictions = loaded_model.predict(x_test_point)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
