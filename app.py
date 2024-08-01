from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
import logging

# Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Load pre-trained model and preprocessing pipeline
with open('voting_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('preprocess_pipeline.pkl', 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input data from the request
        data = request.get_json()

        df = pd.DataFrame([data])

        #Check if required columns are present
        if not all(col in df.columns for col in df.columns):
            return jsonify({'error': 'Invalid input data'}), 400

        # Calculate BMI
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        print(df)

        # Preprocess the input data
        input_data_processed = pipeline.transform(df)
        print(input_data_processed)

        logging.debug(f"Preprocessed data shape: {input_data_processed.shape}")

        # Make prediction
        prediction = model.predict(input_data_processed)
        logging.debug(f"Model prediction: {prediction}")

        # Return prediction result
        result = int(prediction[0])
        logging.debug(f"Returning result: {result}")
        return jsonify({'result': result}), 200

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)