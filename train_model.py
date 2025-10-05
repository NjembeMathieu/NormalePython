# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Charger les données
df = pd.read_csv("Car_Purchasing_Data.csv")

# Préparation des données
features = ['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
X = df[features]
y = df['Car Purchase Amount']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Entraînement et comparaison des modèles
models = {
    'SVM': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

best_score = 0
best_model = None
best_model_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    print(f"{name} R²: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"\nMeilleur modèle: {best_model_name} avec R²: {best_score:.4f}")

# Sauvegarde du meilleur modèle et du scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Modèle et scaler sauvegardés!")