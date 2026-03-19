from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)
# Load the model (ensure you upload model.pkl later)
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        weight = float(data['weight'])
        height = float(data['height'])
        # bmi = weight / (height/100)**2 
        if model:
            # Assuming model takes [weight, height]
            prediction = model.predict(np.array([[weight, height]]))[0]
            return jsonify({'bmi_category': prediction})
        else:
            # Fallback logic if model.pkl isn't uploaded yet
            bmi = round(weight / (height/100)**2, 2)
            return jsonify({'bmi': bmi, 'note': 'Model file not found, using formula.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
