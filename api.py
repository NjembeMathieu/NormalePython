# api.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Charger le modèle sauvegardé
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        # Extraire les features
        features = [
            data['age'],
            data['annual_salary'],
            data['credit_card_debt'],
            data['net_worth']
        ]
        
        # Charger le modèle et le scaler
        model, scaler = load_model()
        
        if model is None or scaler is None:
            return jsonify({'error': 'Modèle non disponible'}), 500
        
        # Prétraitement
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)