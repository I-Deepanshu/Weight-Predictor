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

        # Extract height from the received JSON data
        height = data.get('height', None)

        if height is None:
            return jsonify({'error': 'Height is required'}), 400

        # Make the prediction using the model
        prediction = model.predict(np.array([[height]]))

        # Return the predicted weight as a response
        return jsonify({'predicted_weight': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
