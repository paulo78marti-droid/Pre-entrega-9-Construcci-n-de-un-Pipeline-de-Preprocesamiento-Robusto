"""
Pre-entrega: Construcción de un Pipeline de Preprocesamiento Robusto
Curso: Data Science II - Coderhouse
Dataset: Ames Housing (House Prices)

Este script implementa un pipeline completo de Machine Learning con Scikit-Learn:
1. Carga del conjunto de datos Ames Housing.
2. Partición Train/Test previa para evitar fuga de datos (Data Leakage).
3. Definición de transformadores numéricos y categóricos con ColumnTransformer.
4. Ensamblado de un Pipeline end-to-end con un modelo de regresión (Ridge).
5. Entrenamiento, inferencia y cálculo de métricas (R2, MAE, RMSE).
6. Persistencia del modelo entrenado usando Joblib y prueba con datos crudos.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def load_data():
    """Carga el dataset Ames Housing desde OpenML."""
    print(">>> 1. Cargando dataset Ames Housing (house_prices)...")
    ames_data = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    df = ames_data.frame
    print(f"Dataset cargado con éxito. Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas.")
    return df


def split_data(df, target_col="SalePrice", test_size=0.20, random_state=42):
    """
    Separa variables predictoras (X) de la variable objetivo (y),
    y ejecuta train_test_split ANTES de cualquier transformación (Previene Data Leakage).
    """
    print("\n>>> 2. Separando variables y realizando Train/Test Split...")
    X = df.drop(columns=[target_col, "Id"], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Set de Entrenamiento (Train): {X_train.shape[0]} registros.")
    print(f"Set de Prueba (Test):         {X_test.shape[0]} registros.")
    return X_train, X_test, y_train, y_test


def build_pipeline(X_train):
    """
    Construye el ColumnTransformer y el Pipeline final.
    """
    print("\n>>> 3. Identificando tipos de columnas y construyendo Transformers...")
    # Identificación programática de columnas
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"- Columnas numéricas identificadas:   {len(numeric_features)}")
    print(f"- Columnas categóricas identificadas: {len(categorical_features)}")

    # 1. Pipeline para variables numéricas (Imputación por mediana + Estandarización)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Pipeline para variables categóricas (Imputación por moda + OneHotEncoding seguro)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. Ensamblador ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    # 4. Pipeline Final End-to-End (Preprocesamiento + Estimador Ridge)
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=42))
    ])

    return model_pipeline


def train_and_evaluate(model_pipeline, X_train, y_train, X_test, y_test):
    """
    Entrena el pipeline completo y evalúa su desempeño en el conjunto de test.
    """
    print("\n>>> 4. Entrenando el pipeline completo (Preprocesamiento + Ridge)...")
    model_pipeline.fit(X_train, y_train)
    print("Entrenamiento finalizado con éxito.")

    print("\n>>> 5. Evaluando el modelo en el conjunto de prueba (Test)...")
    y_pred = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("------------------------------------------")
    print("           REPORTE DE MÉTRICAS            ")
    print("------------------------------------------")
    print(f"R² Score (Coef. de Determinación): {r2:.4f}")
    print(f"MAE (Error Absoluto Medio):        ${mae:,.2f} USD")
    print(f"RMSE (Raíz del Error Cuadrático):  ${rmse:,.2f} USD")
    print("------------------------------------------")
    return model_pipeline


def export_and_test_inference(model_pipeline, X_test, filename="ames_housing_pipeline.joblib"):
    """
    Exporta el pipeline entrenado con Joblib y prueba inferencia con datos crudos.
    """
    print(f"\n>>> 6. Persistiendo el pipeline en disco como '{filename}'...")
    joblib.dump(model_pipeline, filename)
    print("Archivo guardado exitosamente.")

    print("\n>>> 7. Verificando carga del pipeline e inferencia con datos crudos...")
    loaded_pipeline = joblib.load(filename)
    test_sample = X_test.iloc[:5]
    sample_preds = loaded_pipeline.predict(test_sample)

    print("Predicciones de prueba generadas directamente sobre datos crudos:")
    for idx, (real_idx, pred_val) in enumerate(zip(test_sample.index, sample_preds)):
        print(f"  Casa ID {real_idx}: Predicción = ${pred_val:,.2f} USD")
    print("\n¡Verificación completada! El pipeline es 100% funcional y robusto.")


if __name__ == "__main__":
    # Flujo de ejecución principal
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline(X_train)
    trained_pipeline = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    export_and_test_inference(trained_pipeline, X_test)
