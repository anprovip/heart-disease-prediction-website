from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('NN.h5', compile=False)

# Initialize the StandardScaler
ss = StandardScaler()

# Define the expected features for scaling
expected_features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Check if all expected features are present in the input
    if not all(feature in data for feature in expected_features):
        return jsonify({'error': 'Missing input data'}), 400
    
    # Extract data
    input_data = np.array([[data[feature] for feature in expected_features]])
    
    # Scale the input data
    scaled_data = ss.fit_transform(input_data)  # Ensure scaler is fitted with same parameters used during training

    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_label = np.argmax(prediction, axis=1)[0]

    # Convert prediction to "Có" or "Không"
    prediction_result = "Có" if prediction_label == 1 else "Không"

    # Output the prediction as JSON
    return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
