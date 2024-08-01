from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# Load pre-trained model and preprocessing pipeline
with open('voting_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define preprocessing pipeline
num_features = ['age', 'bmi', 'ap_hi', 'ap_lo']
cat_features = ['cholesterol', 'gluc', 'gender']

num_pipeline = Pipeline([
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder()),
])

preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input data from the request
        data = request.get_json()
        df = pd.DataFrame([data])

        # Check if required columns are present
        required_columns = num_features + cat_features
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Invalid input data'}), 400

        # Calculate BMI
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

        # Drop columns not used for prediction
        df = df[['age', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'gender']]

        # Preprocess the input data
        input_data_processed = preprocess_pipeline.fit_transform(df)

        # Make prediction
        prediction = model.predict(input_data_processed)

        # Return prediction result
        result = 'Có nguy cơ bệnh tim' if prediction[0] == 1 else 'Không có nguy cơ bệnh tim'
        return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
