import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model import BreastCancerModel

# Initialize Application and Diagnostic Engine
app = Flask(__name__)
cancer_detector = BreastCancerModel()

def initialize_system():
    """Validates model availability or triggers fresh training."""
    if not cancer_detector.load_model():
        print("Required weights not found. Running dataset training...")
        from model import train_and_save_model
        train_and_save_model()
        cancer_detector.load_model()

# Execute startup routine
initialize_system()

@app.route('/')
def home_page():
    """Renders the main diagnostic dashboard."""
    return render_template(
        'index.html', 
        feature_names=cancer_detector.feature_names
    )

@app.route('/predict', methods=['POST'])
def handle_inference():
    """Handles POST requests to classify tumor data."""
    try:
        data_input = request.get_json()
        validated_metrics = {}

        # Validate presence and quality of input attributes
        for field in cancer_detector.feature_names:
            if field not in data_input:
                return jsonify({'success': False, 'message': f'Missing: {field}'}), 400
            
            try:
                val = float(data_input[field])
                if val < 0: 
                    raise ValueError
                validated_metrics[field] = val
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': f'Invalid numeric data: {field}'}), 400

        # Generate prediction via the model wrapper
        predicted_class, probability = cancer_detector.predict(validated_metrics)

        # Logic: 1 = Benign, 0 = Malignant
        is_healthy = bool(predicted_class == 1)
        score = probability if is_healthy else (1.0 - probability)
        
        message = (
            "Analysis complete: The tumor is likely BENIGN." if is_healthy 
            else "Analysis complete: The tumor is likely MALIGNANT."
        )

        return jsonify({
            'status': 'success',
            'output': {
                'label': 'Benign' if is_healthy else 'Malignant',
                'is_benign': is_healthy,
                'confidence': round(score * 100, 2),
                'description': message,
                'disclaimer': 'For academic purposes only. Seek medical advice for diagnosis.'
            }
        })

    except Exception as error_log:
        return jsonify({'status': 'error', 'details': str(error_log)}), 500

@app.route('/fetch-samples')
def provide_example_cases():
    """Grabs random instances from the source data for front-end demonstration."""
    try:
        path_to_csv = os.path.join(os.path.dirname(__file__), 'data', 'breast_cancer.csv')
        dataset = pd.read_csv(path_to_csv)

        # Extraction logic for diverse cases
        sample_benign = dataset[dataset['diagnosis'] == 1].sample(n=1).drop(columns=['diagnosis']).iloc[0].to_dict()
        sample_malignant = dataset[dataset['diagnosis'] == 0].sample(n=1).drop(columns=['diagnosis']).iloc[0].to_dict()

        return jsonify({
            'success': True,
            'payload': {
                'benign_case': sample_benign,
                'malignant_case': sample_malignant
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Start the Flask production-lite server
    app.run(host='0.0.0.0', port=5001, debug=False)