from flask import Flask, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('features.json', 'r') as f:
    features = json.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Salary Predictor API',
        'endpoint': '/predict',
        'method': 'POST'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['work_year', 'experience_level', 'employment_type',
                          'job_title', 'employee_residence', 'remote_ratio',
                          'company_location', 'company_size']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Encode categorical fields
        categorical_cols = ['experience_level', 'employment_type',
                           'job_title', 'employee_residence',
                           'company_location', 'company_size']

        encoded = {}
        for col in categorical_cols:
            value = data[col]
            if value not in encoders[col].classes_:
                return jsonify({'error': f'Unknown value for {col}: {value}'}), 400
            encoded[col] = int(encoders[col].transform([value])[0])

        # Build feature vector in correct order
        input_features = []
        for feature in features:
            if feature in categorical_cols:
                input_features.append(encoded[feature])
            else:
                input_features.append(data[feature])

        # Make prediction
        prediction = model.predict([input_features])[0]

        return jsonify({
            'predicted_salary_usd': round(float(prediction), 2),
            'input': data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/experience-levels', methods=['GET'])
def experience_levels():
    return jsonify({'experience_levels': list(encoders['experience_level'].classes_)})

@app.route('/company-sizes', methods=['GET'])
def company_sizes():
    return jsonify({'company_sizes': list(encoders['company_size'].classes_)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)