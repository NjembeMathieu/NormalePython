# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Car Purchase Predictor",
    page_icon="🚗",
    layout="wide"
)

# Titre de l'application
st.title("🚗 Prédiction du Montant d'Achat de Voiture")
st.markdown("---")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("Car_Purchasing_Data.csv")

# Préprocessing des données
def preprocess_data(df):
    # Sélection des features et target
    features = ['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
    X = df[features]
    y = df['Car Purchase Amount']
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, features

# Entraînement des modèles
def train_models(X_train, y_train):
    # SVM
    svm_model = SVR(kernel='rbf')
    svm_model.fit(X_train, y_train)
    
    # KNN
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    return svm_model, knn_model

# Évaluation des modèles
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': y_pred
    }

# Optimisation du meilleur modèle
def optimize_model(best_model_name, X_train, y_train):
    if best_model_name == "SVM":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear']
        }
        model = SVR()
    else:  # KNN
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        model = KNeighborsRegressor()
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# Volet 1: Interface de prédiction avec modèle local
def local_prediction_interface():
    st.header("🔮 Prédiction Locale avec Modèles ML")
    
    # Chargement des données
    df = load_data()
    
    # Sidebar pour la saisie des données
    st.sidebar.header("📊 Entrez les données du client")
    
    age = st.sidebar.slider("Âge", 18, 80, 40)
    annual_salary = st.sidebar.number_input("Salaire Annuel ($)", 20000, 200000, 60000)
    credit_card_debt = st.sidebar.number_input("Dette Carte de Crédit ($)", 0, 30000, 10000)
    net_worth = st.sidebar.number_input("Valeur Nette ($)", 0, 1000000, 300000)
    
    # Préparation des données
    X, y, scaler, features = preprocess_data(df)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraînement des modèles
    svm_model, knn_model = train_models(X_train, y_train)
    
    # Évaluation des modèles
    svm_results = evaluate_model(svm_model, X_test, y_test, "SVM")
    knn_results = evaluate_model(knn_model, X_test, y_test, "KNN")
    
    # Affichage des performances
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance des Modèles")
        results_df = pd.DataFrame([svm_results, knn_results])
        results_df = results_df[['model', 'rmse', 'r2']]
        st.dataframe(results_df.style.format({
            'rmse': '{:.2f}',
            'r2': '{:.4f}'
        }))
    
    with col2:
        st.subheader("🎯 Meilleur Modèle")
        best_model = max([svm_results, knn_results], key=lambda x: x['r2'])
        st.success(f"**{best_model['model']}** - R²: {best_model['r2']:.4f}")
        
        # Optimisation du meilleur modèle
        if st.button("🔄 Optimiser le Meilleur Modèle"):
            with st.spinner("Optimisation en cours..."):
                optimized_model, best_params = optimize_model(
                    best_model['model'], X_train, y_train
                )
                
                # Évaluation du modèle optimisé
                y_pred_opt = optimized_model.predict(X_test)
                r2_opt = r2_score(y_test, y_pred_opt)
                
                st.success(f"Modèle optimisé - R²: {r2_opt:.4f}")
                st.write("Meilleurs paramètres:", best_params)
    
    # Prédiction avec les données saisies
    input_data = np.array([[age, annual_salary, credit_card_debt, net_worth]])
    input_scaled = scaler.transform(input_data)
    
    svm_prediction = svm_model.predict(input_scaled)[0]
    knn_prediction = knn_model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("🎯 Prédictions")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        st.metric("Prédiction SVM", f"${svm_prediction:,.2f}")
    
    with pred_col2:
        st.metric("Prédiction KNN", f"${knn_prediction:,.2f}")
    
    with pred_col3:
        avg_prediction = (svm_prediction + knn_prediction) / 2
        st.metric("Moyenne des Prédictions", f"${avg_prediction:,.2f}", 
                 delta=f"${avg_prediction - 50000:,.2f}" if avg_prediction > 50000 else f"-${50000 - avg_prediction:,.2f}")
    
    # Visualisation des prédictions
    st.subheader("📈 Visualisation des Prédictions")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique des prédictions vs valeurs réelles pour SVM
    ax1.scatter(y_test, svm_results['predictions'], alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Valeurs Réelles')
    ax1.set_ylabel('Prédictions SVM')
    ax1.set_title(f'SVM - R² = {svm_results["r2"]:.4f}')
    
    # Graphique des prédictions vs valeurs réelles pour KNN
    ax2.scatter(y_test, knn_results['predictions'], alpha=0.6)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Valeurs Réelles')
    ax2.set_ylabel('Prédictions KNN')
    ax2.set_title(f'KNN - R² = {knn_results["r2"]:.4f}')
    
    plt.tight_layout()
    st.pyplot(fig)

# Volet 2: Interface API
def api_prediction_interface():
    st.header("🌐 Prédiction via API")
    
    st.info("""
    Cette section simule l'appel à une API déployée. En production, vous remplacerez 
    l'URL par celle de votre API déployée sur Hugging Face Spaces ou Streamlit Cloud.
    """)
    
    # Formulaire de saisie pour l'API
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Données d'Entrée")
        age_api = st.number_input("Âge (API)", min_value=18, max_value=80, value=40, key="age_api")
        salary_api = st.number_input("Salaire Annuel (API)", min_value=20000, max_value=200000, value=60000, key="salary_api")
    
    with col2:
        st.subheader("")
        debt_api = st.number_input("Dette Carte (API)", min_value=0, max_value=30000, value=10000, key="debt_api")
        worth_api = st.number_input("Valeur Nette (API)", min_value=0, max_value=1000000, value=300000, key="worth_api")
    
    # Simulation de l'appel API
    if st.button("🚀 Appeler l'API pour Prédiction"):
        # En production, vous utiliserez une vraie URL d'API
        # Exemple: url = "https://your-model-api.hf.space/predict"
        
        # Pour la démonstration, nous simulons une réponse d'API
        with st.spinner("Appel de l'API en cours..."):
            # Simulation des données de réponse
            import time
            time.sleep(2)  # Simulation du temps de réponse
            
            # Dans une vraie implémentation, vous feriez:
            # response = requests.post(url, json={
            #     "age": age_api,
            #     "annual_salary": salary_api,
            #     "credit_card_debt": debt_api,
            #     "net_worth": worth_api
            # })
            # prediction = response.json()["prediction"]
            
            # Simulation de prédiction
            simulated_prediction = (age_api * 100 + salary_api * 0.5 + worth_api * 0.1 - debt_api * 0.8)
            
            st.success("✅ Prédiction reçue de l'API!")
            
            # Affichage du résultat
            st.metric(
                "Montant d'Achat Prédit via API", 
                f"${simulated_prediction:,.2f}",
                delta=f"${simulated_prediction - 50000:,.2f}" if simulated_prediction > 50000 else f"-${50000 - simulated_prediction:,.2f}"
            )
            
            # Code d'exemple pour l'appel API réel
            st.subheader("💻 Code d'Exemple pour Appel API")
            st.code(f"""
import requests
import json

# Données d'entrée
data = {{
    "age": {age_api},
    "annual_salary": {salary_api},
    "credit_card_debt": {debt_api},
    "net_worth": {worth_api}
}}

# Appel API (remplacez par votre URL réelle)
url = "https://votre-modele-api.hf.space/predict"
response = requests.post(url, json=data)

if response.status_code == 200:
    prediction = response.json()["prediction"]
    print(f"Prédiction: {{prediction}}")
else:
    print("Erreur API:", response.status_code)
            """)

# Navigation principale
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choisissez le mode d'utilisation",
        ["🔮 Prédiction Locale", "🌐 Prédiction API"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Car Purchase Predictor**  
    Utilisez le machine learning pour prédire le montant d'achat de voiture  
    basé sur l'âge, le salaire, la dette et la valeur nette.
    """)
    
    if app_mode == "🔮 Prédiction Locale":
        local_prediction_interface()
    else:
        api_prediction_interface()

if __name__ == "__main__":
    main()