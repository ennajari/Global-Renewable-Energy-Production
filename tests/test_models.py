import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pytest

# Définir le répertoire de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_raw_data():
    """Charge les données brutes depuis le fichier CSV."""
    data_path = os.path.join(BASE_DIR, "data", "raw", "global_renewable_energy_production.csv")
    print(f"Chargement des données depuis : {data_path}")  # Débogage
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    return pd.read_csv(data_path)

def test_load_raw_data():
    """Teste le chargement des données brutes."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame), "Les données doivent être un DataFrame."
    assert not df.empty, "Le DataFrame ne doit pas être vide."

def test_missing_values():
    """Teste la gestion des valeurs manquantes."""
    df = load_raw_data()
    missing_values = df.isnull().sum().sum()
    assert missing_values == 0, "Il ne doit pas y avoir de valeurs manquantes dans le dataset."

def test_feature_engineering():
    """Teste la création de nouvelles fonctionnalités."""
    df = load_raw_data()
    
    # Exemple de création de nouvelles fonctionnalités
    df['TotalRenewableEnergyPerCapita'] = df['TotalRenewableEnergy'] / 1_000_000
    df['SolarToWindRatio'] = df['SolarEnergy'] / (df['WindEnergy'] + 1e-6)
    
    assert 'TotalRenewableEnergyPerCapita' in df.columns, "La fonctionnalité 'TotalRenewableEnergyPerCapita' doit être créée."
    assert 'SolarToWindRatio' in df.columns, "La fonctionnalité 'SolarToWindRatio' doit être créée."

def test_data_preprocessing():
    """Teste le prétraitement des données."""
    df = load_raw_data()
    
    # Définir les colonnes numériques et catégorielles
    numerical_features = ['SolarEnergy', 'WindEnergy', 'HydroEnergy', 'OtherRenewableEnergy', 'TotalRenewableEnergy']
    categorical_features = ['Country']
    
    # Pipeline pour les colonnes numériques
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour les colonnes catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combiner les pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Appliquer le prétraitement
    df_processed = preprocessor.fit_transform(df)
    
    assert df_processed.shape[0] == df.shape[0], "Le nombre de lignes ne doit pas changer après le prétraitement."
    assert df_processed.shape[1] > df.shape[1], "Le nombre de colonnes doit augmenter après le prétraitement."

# Exécuter les tests
if __name__ == "__main__":
    pytest.main()