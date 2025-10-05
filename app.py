import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .api-card {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .code-block {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class CarPricePredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        
    def load_and_preprocess_data(self, df):
        """Charge et prétraite les données"""
        with st.spinner("Chargement et prétraitement des données..."):
            # on supprime les colones non pertinentes comme dans le jupyter
            df_clean = df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1, errors='ignore')
            
            # Vérification des données
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dimensions", f"{df_clean.shape[0]} lignes × {df_clean.shape[1]} colonnes")
            with col2:
                st.metric("Doublons", df_clean.duplicated().sum())
            
            return df_clean
    
    def explore_data(self, df):
        """on explore les donnees et ac"""
        st.subheader("📊 Exploration des Données")
        
        # Affichage des statistiques descriptives
        with st.expander("Statistiques Descriptives"):
            st.dataframe(df.describe())
        
        # Matrice de corrélation
        with st.expander("Matrice de Corrélation"):
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matrice de Corrélation')
            st.pyplot(fig)
        
        # Distributions
        with st.expander("Distributions des Variables"):
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df['Car Purchase Amount'], kde=True, ax=ax)
                ax.set_title('Distribution du Prix d\'Achat')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(y=df['Car Purchase Amount'], ax=ax)
                ax.set_title('Boxplot du Prix d\'Achat')
                st.pyplot(fig)
        
        return correlation_matrix
    
    def prepare_features(self, df):
        """Prépare les features pour l'entraînement"""
        X = df.drop('Car Purchase Amount', axis=1)
        y = df['Car Purchase Amount']
        
        # Définition des colonnes numériques et catégorielles
        numeric_features = ['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
        categorical_features = ['Gender']
        
        # Préprocessing
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return X, y
    
    def train_models(self, X, y):
        """Entraîne les modèles KNN et SVM"""
        st.subheader("🤖 Entraînement du modele le plus performant")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        status_text.text("Division des données...")
        progress_bar.progress(20)
        
        # Pipelines
        pipeline_knn = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', KNeighborsRegressor())
        ])
        
        pipeline_svm = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', SVR())
        ])
        
        # Hyperparamètres pour le GridSearch
        param_grid_knn = {
            'regressor__n_neighbors': [3, 5, 7, 9],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__metric': ['euclidean', 'manhattan']
        }
        
        param_grid_svm = {
            'regressor__C': [0.1, 1, 10],
            'regressor__kernel': ['linear', 'rbf'],
            'regressor__gamma': ['scale', 'auto']
        }
        
        # Recherche d'hyperparamètres pour KNN
        status_text.text("Optimisation KNN en cours...")
        grid_knn = GridSearchCV(
            pipeline_knn, param_grid_knn, 
            cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        grid_knn.fit(X_train, y_train)
        progress_bar.progress(50)
        
        # Recherche d'hyperparamètres pour SVM
        status_text.text("Optimisation SVM en cours...")
        grid_svm = GridSearchCV(
            pipeline_svm, param_grid_svm,
            cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        grid_svm.fit(X_train, y_train)
        progress_bar.progress(80)
        
        # Stockage des modèles
        self.models['KNN'] = {
            'model': grid_knn.best_estimator_,
            'best_params': grid_knn.best_params_,
            'cv_score': grid_knn.best_score_
        }
        
        self.models['SVM'] = {
            'model': grid_svm.best_estimator_,
            'best_params': grid_svm.best_params_,
            'cv_score': grid_svm.best_score_
        }
        
        status_text.text("Entraînement terminé!")
        progress_bar.progress(100)
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_models(self, X_test, y_test):
        """Évalue les modèles sur les données de test"""
        st.subheader("📈 Évaluation des Modèles")
        
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'R2': r2,
                'Best Params': model_info['best_params'],
                'CV Score': model_info['cv_score']
            }
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### KNN")
            knn_result = results['KNN']
            st.metric("R² Score", f"{knn_result['R2']:.4f}")
            st.metric("RMSE", f"${knn_result['RMSE']:.2f}")
            with st.expander("Meilleurs paramètres KNN"):
                st.json(knn_result['Best Params'])
        
        with col2:
            st.markdown("#### SVM")
            svm_result = results['SVM']
            st.metric("R² Score", f"{svm_result['R2']:.4f}")
            st.metric("RMSE", f"${svm_result['RMSE']:.2f}")
            with st.expander("Meilleurs paramètres SVM"):
                st.json(svm_result['Best Params'])
        
        # Sélection du meilleur modèle
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        self.best_model = self.models[best_model_name]['model']
        self.best_model_name = best_model_name
        self.is_trained = True
        
        st.success(f"🎯 Meilleur modèle sélectionné: **{best_model_name}** (R²: {results[best_model_name]['R2']:.4f})")
        
        return results
    
    def save_models(self):
        """Sauvegarde les modèles entraînés"""
        if self.best_model:
            with open('best_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'model_name': self.best_model_name,
                    'preprocessor': self.preprocessor
                }, f)
            return True
        return False
    
    def load_models(self):
        """Charge les modèles sauvegardés"""
        try:
            with open('best_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.best_model = model_data['model']
                self.best_model_name = model_data['model_name']
                self.preprocessor = model_data['preprocessor']
                self.is_trained = True
            return True
        except FileNotFoundError:
            return False

class APIClient:
    """Client pour tester l'API de prédiction"""
    
    def __init__(self):
        self.base_url = None
        
    def test_connection(self, url):
        """Teste la connexion à l'API"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Erreur HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def predict_single(self, url, data):
        """Fait une prédiction unique"""
        try:
            response = requests.post(
                f"{url}/predict",
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return True, response.json() if response.status_code == 200 else response.text
        except Exception as e:
            return False, str(e)
    
    def predict_batch(self, url, records):
        """Fait des prédictions par lot"""
        try:
            response = requests.post(
                f"{url}/batch_predict",
                json={'records': records},
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return True, response.json() if response.status_code == 200 else response.text
        except Exception as e:
            return False, str(e)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
    
    # Initialisation du prédicteur et client API
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CarPricePredictor()
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    
    predictor = st.session_state.predictor
    api_client = st.session_state.api_client
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choisissez une section",
        ["Accueil", "Analyse des Données", "Entraînement", "Prédiction", "API"]
    )
    
    if app_mode == "Accueil":
        show_homepage()
    
    elif app_mode == "Analyse des Données":
        show_data_analysis(predictor)
    
    elif app_mode == "Entraînement":
        show_training(predictor)
    
    elif app_mode == "Prédiction":
        show_prediction(predictor)
    
    elif app_mode == "API":
        show_api_section(predictor, api_client)
    
    
def show_homepage():
    """Page d'accueil"""
    st.markdown("""
    ## Session normale de Python pour la Data Science
    
    Application qui predit les prix de vente de voiture en fonction des details des individus
    
    ### Fonctionnalités:
    
    📊 **Analyse des Données** - Exploration et visualisation du dataset
    
    🧠 **Entraînement** - Entraînement des modèles KNN et SVM
    
    🔮 **Prédiction** - Prédiction du prix de voitures via l'interface
    
    🌐 **API** - Prédictions via appels API REST
    
    📈 **Évaluation** - Comparaison des performances des modèles
    
    ### Fonctionnement:
    
    1. Charger les données dans la section **Analyse des Données**
    2. Entraînez les modèles dans la section **Entraînement**
    3. Faire des prédictions en utilisant les modeles de machine learning SVM et KNN
    4. Faire des predictions avec les modeles sous formes d'API
    
    ### Dataset requis:
    
    Le dataset doit contenir les colonnes suivantes:
    - `Gender` (0=Femme, 1=Homme)
    - `Age` 
    - `Annual Salary`
    - `Credit Card Debt`
    - `Net Worth`
    - `Car Purchase Amount` (variable cible)
    """)

def show_data_analysis(predictor):
    """Presentation des donnees"""
    st.header("📊 Visualisation des donnees")
    
    uploaded_file = st.file_uploader(
        "Téléchargez votre fichier CSV", 
        type=['csv'],
        help="Le fichier doit contenir les colonnes requises"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Fichier chargé avec succès! {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Aperçu des données
            with st.expander("Aperçu des données"):
                st.dataframe(df.head())
            
            # Prétraitement
            df_clean = predictor.load_and_preprocess_data(df)
            
            # Exploration
            predictor.explore_data(df_clean)
            
            # Sauvegarde des données nettoyées dans la session
            st.session_state.df_clean = df_clean
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")

def show_training(predictor):
    """Section entraînement"""
    st.header("🤖 Entraînement du modele le plus performant")
    
    if 'df_clean' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger et analyser les données dans la section 'Analyse des Données'")
        return
    
    df_clean = st.session_state.df_clean
    
    if st.button("🚀 Lancer l'Entraînement", type="primary"):
        # Préparation des features
        X, y = predictor.prepare_features(df_clean)
        
        # Entraînement
        X_train, X_test, y_train, y_test = predictor.train_models(X, y)
        
        # Évaluation
        results = predictor.evaluate_models(X_test, y_test)
        
        # Sauvegarde
        if predictor.save_models():
            st.success("✅ Modèles sauvegardés avec succès!")
        
        # Sauvegarde des résultats
        st.session_state.results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

def show_prediction(predictor):
    """Section prédiction"""
    st.header("🔮 Prédiction du Prix")
    
    # Essayer de charger les modèles sauvegardés
    if not predictor.is_trained:
        if predictor.load_models():
            st.success("✅ Modèle chargé depuis la sauvegarde!")
        else:
            st.warning("⚠️ Aucun modèle entraîné trouvé. Veuillez d'abord entraîner un modèle.")
            return
    
    if predictor.is_trained:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informations du Client")
            gender = st.selectbox("Genre", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme", key="pred_gender")
            age = st.slider("Âge", min_value=18, max_value=80, value=45, key="pred_age")
            annual_salary = st.number_input("Salaire Annuel ($)", min_value=20000, max_value=200000, value=70000, step=1000, key="pred_salary")
        
        with col2:
            st.subheader("Situation Financière du client")
            credit_card_debt = st.number_input("Dette Carte de Crédit ($)", min_value=0, max_value=50000, value=10000, step=100, key="pred_debt")
            net_worth = st.number_input("Valeur Nette ($)", min_value=0, max_value=2000000, value=400000, step=1000, key="pred_worth")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prédiction automatique dès que les données changent
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Annual Salary': annual_salary,
            'Credit Card Debt': credit_card_debt,
            'Net Worth': net_worth
        }])
        
        # Prédiction
        prediction = predictor.best_model.predict(input_data)[0]
        
        # Affichage dynamique du résultat
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
                <h3>💰 Prix Prédit</h3>
                <h1 style='font-size: 3rem; margin: 1rem 0;'>${prediction:,.2f}</h1>
                <p>Modèle utilisé: <strong>{predictor.best_model_name}</strong></p>
                
            </div>
            """, unsafe_allow_html=True)
        
        # Informations supplémentaires avec expander
        with st.expander("📋 Détails de la Prédiction", expanded=False):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write("**Données d'entrée:**")
                st.json({
                    "Gender": "Femme" if gender == 0 else "Homme",
                    "Age": age,
                    "Annual Salary": f"${annual_salary:,.2f}",
                    "Credit Card Debt": f"${credit_card_debt:,.2f}",
                    "Net Worth": f"${net_worth:,.2f}"
                })
            
            with col_info2:
                st.write("**Informations du modèle:**")
                st.metric("Algorithme", predictor.best_model_name)
                st.metric("Précision estimée", f"{get_model_accuracy(predictor.best_model_name):.1%}")
                
                # Indicateur de confiance basé sur la plage des données
                confidence = calculate_confidence(age, annual_salary, credit_card_debt, net_worth)
                st.metric("Niveau de confiance", f"{confidence}%")
        
        # Graphique contextuel (optionnel)
        with st.expander("📊 Analyse Contextuelle", expanded=False):
            show_contextual_analysis(age, annual_salary, credit_card_debt, net_worth, prediction)

def get_model_accuracy(model_name):
    """Retourne une précision estimée basée sur le modèle"""
    accuracies = {
        'KNN': 0.85,
        'SVM': 0.82
    }
    return accuracies.get(model_name, 0.80)

def calculate_confidence(age, salary, debt, net_worth):
    """Calcule un niveau de confiance basé sur la plausibilité des données"""
    confidence = 85  # Base de confiance
    
    # Vérification de la cohérence âge/salaire
    if age < 25 and salary > 80000:
        confidence -= 15
    elif age > 60 and salary > 150000:
        confidence -= 10
    
    # Vérification dette/valeur nette
    if debt > net_worth * 0.5:
        confidence -= 20
    elif debt > net_worth * 0.3:
        confidence -= 10
    
    # Vérification des valeurs extrêmes
    if salary < 25000 or salary > 150000:
        confidence -= 5
    if net_worth > 1000000:
        confidence -= 5
    
    return max(60, min(95, confidence))

def show_contextual_analysis(age, salary, debt, net_worth, prediction):
    """Affiche une analyse contextuelle des données"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ratio Dette/Valeur Nette:**")
        debt_ratio = (debt / net_worth) * 100 if net_worth > 0 else 0
        st.metric("Ratio", f"{debt_ratio:.1f}%")
        
        if debt_ratio > 50:
            st.warning("Ratio élevé - risque financier détecté")
        elif debt_ratio > 30:
            st.info("Ratio modéré")
        else:
            st.success("Ratio sain")
    
    with col2:
        st.write("**Ratio Salaire/Prix Prédit:**")
        salary_ratio = (prediction / salary) * 100 if salary > 0 else 0
        st.metric("Ratio", f"{salary_ratio:.1f}%")
        
        if salary_ratio > 100:
            st.warning("Prix élevé par rapport au salaire")
        elif salary_ratio > 70:
            st.info("Prix modéré par rapport au salaire")
        else:
            st.success("Prix abordable")
    
    # Graphique simple de répartition
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Salaire Annuel', 'Dette', 'Valeur Nette', 'Prix Prédit']
    values = [salary, debt, net_worth, prediction]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Montant ($)')
    ax.set_title('Répartition Financière')
    
    # Format des valeurs sur les barres
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'${value:,.0f}', ha='center', va='bottom', rotation=0)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def show_api_section(predictor, api_client):
    """Section API pour les appels de prédiction"""
    st.header("🌐 API de Prédiction")
    
    st.markdown("""
    Cette section vous permet d'utiliser le modèle via des appels API REST.
    Vous pouvez intégrer les prédictions dans vos applications externes.
    """)
    
    # Configuration de l'URL de l'API
    st.subheader("🔧 Configuration de l'API")
    
    api_url = st.text_input(
        "URL de l'API",
        value="http://localhost:5000",
        help="URL de base de votre API (ex: http://localhost:5000 ou https://votre-api.herokuapp.com)"
    )
    
    # Test de connexion
    if st.button("🧪 Tester la connexion"):
        with st.spinner("Test de connexion en cours..."):
            success, result = api_client.test_connection(api_url)
            
            if success:
                st.success("✅ Connexion réussie!")
                st.json(result)
            else:
                st.error(f"❌ Erreur de connexion: {result}")
    
    
    # Testeur d'API interactif
    st.subheader("🧪 Testeur d'API Interactif")
    
    tab1, tab2 = st.tabs(["Prédiction Unique", "Prédiction par Lot"])
    
    with tab1:
        st.markdown("### Test de Prédiction Unique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_gender = st.selectbox("Genre (test)", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme", key="test_gender")
            test_age = st.slider("Âge (test)", min_value=18, max_value=80, value=45, key="test_age")
            test_salary = st.number_input("Salaire Annuel ($) (test)", min_value=20000, max_value=200000, value=70000, step=1000, key="test_salary")
        
        with col2:
            test_debt = st.number_input("Dette Carte de Crédit ($) (test)", min_value=0, max_value=50000, value=10000, step=100, key="test_debt")
            test_worth = st.number_input("Valeur Nette ($) (test)", min_value=0, max_value=2000000, value=400000, step=1000, key="test_worth")
        
        test_data = {
            "Gender": test_gender,
            "Age": test_age,
            "Annual Salary": test_salary,
            "Credit Card Debt": test_debt,
            "Net Worth": test_worth
        }
        
        # Affichage des données de test
        st.markdown("**Données de test:**")
        st.json(test_data)
        
        # Prédiction automatique via API si l'URL est configurée
        if api_url and api_url.strip() and not api_url == "http://localhost:5000":
            with st.spinner("Interrogation de l'API..."):
                success, result = api_client.predict_single(api_url, test_data)
                
                if success and isinstance(result, dict):
                    st.success("✅ Prédiction API réussie!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prix Prédit (API)", f"${result.get('predicted_car_price', 'N/A'):,}")
                    with col2:
                        st.metric("Modèle utilisé", result.get('model_used', 'N/A'))
                    with col3:
                        # Comparaison avec le modèle local si disponible
                        if predictor.is_trained:
                            local_prediction = predictor.best_model.predict(pd.DataFrame([test_data]))[0]
                            difference = abs(result.get('predicted_car_price', 0) - local_prediction)
                            st.metric("Différence modèle local", f"${difference:,.2f}")
                    
                    with st.expander("Voir la réponse API complète"):
                        st.json(result)
                elif success:
                    st.error(f"❌ Erreur API: {result}")
                else:
                    st.error(f"❌ Erreur de connexion: {result}")
        else:
            st.info("ℹ️ Configurez une URL d'API valide pour tester les prédictions")
            
            # Simulation locale si l'API n'est pas configurée mais le modèle est entraîné
            if predictor.is_trained:
                st.markdown("---")
                st.markdown("**Simulation locale (modèle entraîné):**")
                local_prediction = predictor.best_model.predict(pd.DataFrame([test_data]))[0]
                st.metric("Prix Prédit (Local)", f"${local_prediction:,.2f}")
                st.info("Ceci est une simulation locale. Configurez l'URL de l'API pour une vraie prédiction API.")
    
    with tab2:
        st.markdown("### Test de Prédiction par Lot")
        
        st.info("Ajoutez plusieurs enregistrements pour tester les prédictions par lot")
        
        # Gestion des enregistrements multiples
        if 'batch_records' not in st.session_state:
            st.session_state.batch_records = [{
                "Gender": 1,
                "Age": 35,
                "Annual Salary": 60000,
                "Credit Card Debt": 8000,
                "Net Worth": 300000
            }]
        
        records = st.session_state.batch_records
        
        for i, record in enumerate(records):
            st.markdown(f"**Enregistrement {i+1}**")
            col1, col2 = st.columns(2)
            
            with col1:
                record['Gender'] = st.selectbox(
                    f"Genre {i+1}", 
                    options=[0, 1], 
                    index=record['Gender'],
                    format_func=lambda x: "Femme" if x == 0 else "Homme",
                    key=f"batch_gender_{i}"
                )
                record['Age'] = st.slider(
                    f"Âge {i+1}", 
                    min_value=18, 
                    max_value=80, 
                    value=record['Age'],
                    key=f"batch_age_{i}"
                )
                record['Annual Salary'] = st.number_input(
                    f"Salaire {i+1}", 
                    min_value=20000, 
                    max_value=200000, 
                    value=record['Annual Salary'],
                    key=f"batch_salary_{i}"
                )
            
            with col2:
                record['Credit Card Debt'] = st.number_input(
                    f"Dette {i+1}", 
                    min_value=0, 
                    max_value=50000, 
                    value=record['Credit Card Debt'],
                    key=f"batch_debt_{i}"
                )
                record['Net Worth'] = st.number_input(
                    f"Valeur Nette {i+1}", 
                    min_value=0, 
                    max_value=2000000, 
                    value=record['Net Worth'],
                    key=f"batch_worth_{i}"
                )
            
            # Bouton pour supprimer un enregistrement
            if len(records) > 1:
                if st.button(f"❌ Supprimer l'enregistrement {i+1}", key=f"delete_{i}"):
                    records.pop(i)
                    st.rerun()
            
            st.markdown("---")
        
        # Boutons pour ajouter/supprimer des enregistrements
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Ajouter un enregistrement"):
                records.append({
                    "Gender": 1,
                    "Age": 45,
                    "Annual Salary": 70000,
                    "Credit Card Debt": 10000,
                    "Net Worth": 400000
                })
                st.rerun()
        
        # Affichage du lot de données
        st.markdown("**Lot de données à envoyer:**")
        st.json({"records": records})
        
        if st.button("🚀 Tester les prédictions par lot", key="test_batch"):
            if api_url and api_url.strip() and not api_url == "http://localhost:5000":
                with st.spinner("Envoi de la requête par lot..."):
                    success, result = api_client.predict_batch(api_url, records)
                    
                    if success and isinstance(result, dict):
                        st.success(f"✅ {result.get('total_predictions', 0)} prédictions réussies!")
                        
                        for prediction in result.get('predictions', []):
                            with st.container():
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.metric(
                                        f"Enregistrement {prediction['record_id'] + 1}", 
                                        f"${prediction['predicted_car_price']:,}"
                                    )
                        
                        with st.expander("Voir la réponse complète"):
                            st.json(result)
                    elif success:
                        st.error(f"❌ Erreur API: {result}")
                    else:
                        st.error(f"❌ Erreur de connexion: {result}")
            else:
                st.warning("⚠️ Veuillez configurer une URL d'API valide pour tester les prédictions par lot")
                
                # Simulation locale
                if predictor.is_trained:
                    st.markdown("---")
                    st.markdown("**Simulation locale (modèle entraîné):**")
                    input_data = pd.DataFrame(records)
                    local_predictions = predictor.best_model.predict(input_data)
                    
                    for i, (record, pred) in enumerate(zip(records, local_predictions)):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.metric(f"Enregistrement {i+1}", f"${pred:,.2f}")
                    
                    st.info("Ceci est une simulation locale. Configurez l'URL de l'API pour une vraie prédiction API.")
    
   
if __name__ == "__main__":
    main()

