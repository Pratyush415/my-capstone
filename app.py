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
        weight = float(data['weight'])
        height = float(data['height'])
        
        if model:
            # Create a DataFrame with the exact column names used during training
            # Note: Ensure 'Weight' and 'Height' match your training script exactly
            features = pd.DataFrame([[weight, height]], columns=['Weight', 'Height'])
            
            prediction = model.predict(features)[0]
            
            # If your model returns an array, extract the first value
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]
                
            return jsonify({'result': f"{prediction:.2f} (Predicted)"})
        else:
            # Fallback logic if model.pkl is missing
            bmi = round(weight / (height / 100) ** 2, 2)
            return jsonify({
                'result': f"{bmi} (Calculated)", 
                'note': 'Model file not found, using standard BMI formula'
            })
            
    except Exception as e:
        # This will help you see the exact error in the browser console if it fails
        return jsonify({'error': str(e)}), 400
