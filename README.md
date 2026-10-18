# Pre-entrega: Construcción de un Pipeline de Preprocesamiento Robusto

**Curso:** Data Science II - Coderhouse  
**Entregable:** Pipeline de preprocesamiento y modelado con Scikit-Learn  

---

## 1. Descripción del Dataset Utilizado

Para esta práctica se utilizó el dataset **Ames Housing** (`house_prices` en OpenML / Kaggle):

* **Descripción:** Contiene información detallada sobre ventas de casas en Ames, Iowa.
* **Dimensiones:** 1.460 filas y 81 columnas.
* **Variable objetivo ($y$):** `SalePrice` (precio final de venta en dólares).
* **Variables predictoras ($X$):**
  * **36 variables numéricas:** áreas en pies cuadrados (`LotArea`, `GrLivArea`, `TotalBsmtSF`), cantidad de baños, habitaciones, año de construcción (`YearBuilt`), etc.
  * **43 variables categóricas:** tipo de vivienda (`BldgType`), barrio (`Neighborhood`), calidad y materiales (`ExterQual`, `Foundation`, etc.).

---

## 2. Construcción del Pipeline

Para cumplir con lo solicitado en la entrega y evitar la **fuga de datos (data leakage)**, se implementó la siguiente estructura:

### A. División Train / Test previa
Antes de calcular cualquier media, mediana o categorías, se dividió el dataset en:
* **80% Entrenamiento (`X_train`, `y_train`):** 1.168 casas.
* **20% Prueba (`X_test`, `y_test`):** 292 casas.

### B. ColumnTransformer (Tratamiento diferenciado)
Se utilizó `ColumnTransformer` para separar las columnas según su tipo y aplicar al menos dos transformaciones a cada una:

1. **Variables Numéricas (`numeric_transformer`):**
   * **Paso 1 (Imputación):** `SimpleImputer(strategy='median')` para rellenar valores nulos con la mediana (más robusta frente a valores atípicos que la media).
   * **Paso 2 (Escalado):** `StandardScaler()` para que todas las variables numéricas tengan la misma escala.

2. **Variables Categóricas (`categorical_transformer`):**
   * **Paso 1 (Imputación):** `SimpleImputer(strategy='most_frequent')` para rellenar nulos con el valor más repetido (moda).
   * **Paso 2 (Codificación):** `OneHotEncoder(handle_unknown='ignore')` para convertir categorías a valores binarios sin que falle si en el futuro aparece una categoría desconocida.

### C. Pipeline Final con Estimador
Se unió el preprocesador (`ColumnTransformer`) y un modelo básico de regresión (`Ridge`) en un único objeto `Pipeline`:
```python
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0, random_state=42))
])
```

---

## 3. Entrenamiento y Métricas Obtenidas

Se entrenó el pipeline completo con `model_pipeline.fit(X_train, y_train)` y se evaluó con los datos de prueba (`X_test`):

* **$R^2$ (Coeficiente de determinación):** `0.8844` (el modelo explica el 88,4% de la variación de los precios).
* **MAE (Error Absoluto Medio):** `$19,096.24 USD` (diferencia promedio entre el precio predicho y el real).
* **RMSE (Raíz del Error Cuadrático Medio):** `$29,777.46 USD`.

---

## 4. Persistencia y Prueba con Nuevos Datos

El pipeline entrenado se exportó a disco usando `joblib`:

```python
import joblib

# Guardar pipeline
joblib.dump(model_pipeline, 'ames_housing_pipeline.joblib')

# Cargar y probar inferencia directa sobre datos crudos sin errores de formato
loaded_pipeline = joblib.load('ames_housing_pipeline.joblib')
predicciones = loaded_pipeline.predict(X_test.iloc[:5])
print("Predicciones:", predicciones)
```

El pipeline cargado recibe datos crudos y devuelve las predicciones directamente sin necesidad de transformar las columnas a mano.

---

## 5. Respuestas a las Preguntas de la Entrega

1. **¿El repositorio contiene un archivo README.md inicial con la descripción del dataset utilizado?**  
   **Sí.** Este archivo describe el dataset Ames Housing, sus dimensiones, atributos y la variable a predecir (`SalePrice`).

2. **¿El código utiliza la clase `Pipeline` de Scikit-Learn para encadenar al menos dos pasos de transformación?**  
   **Sí.** Se utiliza `Pipeline` con 2 pasos en numéricas (imputación con mediana + escalado) y 2 pasos en categóricas (imputación frecuente + One-Hot). Además, el pipeline final une preprocesamiento y estimador.

3. **¿Se implementa un `ColumnTransformer` que diferencia explícitamente el tratamiento de variables numéricas y categóricas?**  
   **Sí.** Se implementa `ColumnTransformer` aplicando transformaciones específicas para las 36 variables numéricas y las 43 categóricas.

4. **¿El código incluye la división de datos (Train/Test) antes de cualquier ajuste de parámetros de preprocesamiento?**  
   **Sí.** `train_test_split` se realiza al inicio para evitar *data leakage* (fuga de datos).

5. **¿El pipeline final es funcional y permite ejecutar el método `.predict()` sobre nuevos datos sin errores de formato?**  
   **Sí.** El pipeline permite ejecutar `.predict()` sobre el conjunto de test y datos nuevos crudos, y se comprobó exportándolo y recargándolo con `joblib`.

---

## 6. Archivos del Repositorio

* `Preentrega_Pipeline_AmesHousing.ipynb`: Notebook con el paso a paso explicado y ejecutado.
* `pipeline_ames_housing.py`: Script en Python ejecutable con todo el pipeline.
* `README.md`: Este archivo explicativo de la entrega.
* `requirements.txt`: Librerías necesarias para ejecutar el código.
