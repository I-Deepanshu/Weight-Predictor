from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        height = data.get('height')

        if height is None:
            return jsonify({'error': 'Height is required'}), 400

        # Predict weight using the model
        prediction = model.predict(np.array([[height]]))
        return jsonify({'predicted_weight': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
