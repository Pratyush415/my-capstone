from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
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
        # 1. Get data from the frontend
        weight = float(data['weight'])
        height = float(data['height'])
        
        if model:
            # 2. CREATE DATAFRAME: This fixes the "Feature Names" error in your logs
            features = pd.DataFrame([[weight, height]], columns=['weight', 'height'])
            
            # 3. Predict using the model
            prediction = model.predict(features)[0]
            
            # Extract value if it's a numpy array
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]
                
            return jsonify({'result': f"{prediction:.2f} (Predicted)"})
        else:
            # Fallback if model.pkl fails to load
            bmi = round(weight / (height / 100) ** 2, 2)
            return jsonify({
                'result': f"{bmi} (Calculated)", 
                'note': 'Model not loaded, using formula'
            })
            
    except Exception as e:
        # This sends the error message back to your browser console for debugging
        return jsonify({'error': str(e)}), 400
