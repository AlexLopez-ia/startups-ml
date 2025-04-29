import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import handle_missing_values, encode_categorical_features, scale_features, preprocess_data

# Datos de ejemplo para los tests
def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, np.nan, 4],
        'cat1': ['a', 'b', 'a', np.nan],
        'num2': [10, 20, 30, 40],
        'target': [0, 1, 0, 1]
    })

def test_handle_missing_values():
    df = sample_df()
    df_proc = handle_missing_values(df, strategy='mean')
    assert df_proc.isnull().sum().sum() == 0, "No debe haber valores nulos tras la imputación"
    assert df_proc['num1'].iloc[2] == pytest.approx(2.333, rel=1e-2)
    assert df_proc['cat1'].iloc[3] == 'a'

def test_encode_categorical_features():
    df = sample_df()
    df_proc = encode_categorical_features(df, target_column='target')
    # Debe haber columnas dummies para cat1
    assert any('cat1_' in col for col in df_proc.columns)
    # El número de filas debe ser el mismo
    assert len(df_proc) == len(df)

def test_scale_features():
    df = sample_df()
    df_proc = scale_features(df, target_column='target')
    # La media de las columnas numéricas (sin target) debe ser ~0
    num_cols = [col for col in df_proc.columns if col.startswith('num')]
    for col in num_cols:
        assert abs(df_proc[col].mean()) < 1e-6

def test_preprocess_data():
    df = sample_df()
    df_proc = preprocess_data(df, target_column='target', scale=True)
    # Debe estar todo preprocesado y sin nulos
    assert df_proc.isnull().sum().sum() == 0
    # Deben existir columnas dummies
    assert any('cat1_' in col for col in df_proc.columns)
    # La media de las columnas numéricas debe ser ~0
    num_cols = [col for col in df_proc.columns if col.startswith('num')]
    for col in num_cols:
        assert abs(df_proc[col].mean()) < 1e-6