from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes

# Chargement du modèle
try:
    with open('best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        model_name = model_data['model_name']
        print(f"Modèle chargé: {model_name}")
except FileNotFoundError:
    print("ERREUR: Le modèle n'a pas été trouvé. Veuillez d'abord exécuter l'entraînement.")
    model = None
    model_name = "Aucun modèle chargé"

@app.route('/')
def home():
    return jsonify({
        'message': 'API de prédiction du prix des voitures',
        'model': model_name,
        'endpoints': {
            '/health': 'GET - Statut de l\'API',
            '/predict': 'POST - Prédire le prix d\'une voiture',
            '/batch_predict': 'POST - Prédictions par lot'
        },
        'documentation': 'Voir /help pour plus d\'informations'
    })

@app.route('/health')
def health():
    status = 'healthy' if model is not None else 'no model loaded'
    return jsonify({'status': status, 'model': model_name})

@app.route('/help')
def help():
    return jsonify({
        'documentation': {
            'input_format': {
                'Gender': 'int (0=Femme, 1=Homme)',
                'Age': 'int',
                'Annual Salary': 'float',
                'Credit Card Debt': 'float',
                'Net Worth': 'float'
            },
            'endpoints': {
                'POST /predict': {
                    'description': 'Prédiction unique',
                    'example_request': {
                        'Gender': 1,
                        'Age': 45,
                        'Annual Salary': 70000,
                        'Credit Card Debt': 10000,
                        'Net Worth': 400000
                    }
                },
                'POST /batch_predict': {
                    'description': 'Prédictions par lot',
                    'example_request': {
                        'records': [
                            {
                                'Gender': 1,
                                'Age': 35,
                                'Annual Salary': 60000,
                                'Credit Card Debt': 8000,
                                'Net Worth': 300000
                            }
                        ]
                    }
                }
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non disponible. Entraînez d\'abord le modèle.'}), 500
    
    try:
        # Récupération des données JSON
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400
        
        # Validation des champs requis
        required_fields = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400
        
        # Validation des types
        try:
            input_data = pd.DataFrame([{
                'Gender': int(data['Gender']),
                'Age': float(data['Age']),
                'Annual Salary': float(data['Annual Salary']),
                'Credit Card Debt': float(data['Credit Card Debt']),
                'Net Worth': float(data['Net Worth'])
            }])
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Type de données invalide: {str(e)}'}), 400
        
        # Prédiction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'predicted_car_price': round(prediction, 2),
            'model_used': model_name,
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': 'Modèle non disponible'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400
        
        records = data.get('records', [])
        
        if not records:
            return jsonify({'error': 'Aucun enregistrement fourni'}), 400
        
        if not isinstance(records, list):
            return jsonify({'error': 'Le champ "records" doit être une liste'}), 400
        
        # Validation des enregistrements
        validated_records = []
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                return jsonify({'error': f'L\'enregistrement {i} doit être un objet JSON'}), 400
            
            required_fields = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
            for field in required_fields:
                if field not in record:
                    return jsonify({'error': f'Champ manquant dans l\'enregistrement {i}: {field}'}), 400
            
            try:
                validated_records.append({
                    'Gender': int(record['Gender']),
                    'Age': float(record['Age']),
                    'Annual Salary': float(record['Annual Salary']),
                    'Credit Card Debt': float(record['Credit Card Debt']),
                    'Net Worth': float(record['Net Worth'])
                })
            except (ValueError, TypeError) as e:
                return jsonify({'error': f'Type de données invalide dans l\'enregistrement {i}: {str(e)}'}), 400
        
        # Conversion en DataFrame
        input_data = pd.DataFrame(validated_records)
        
        # Prédictions
        predictions = model.predict(input_data)
        
        results = []
        for i, (record, pred) in enumerate(zip(records, predictions)):
            results.append({
                'record_id': i,
                'predicted_car_price': round(pred, 2),
                'input_data': record
            })
        
        return jsonify({
            'predictions': results,
            'model_used': model_name,
            'total_predictions': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction par lot: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
