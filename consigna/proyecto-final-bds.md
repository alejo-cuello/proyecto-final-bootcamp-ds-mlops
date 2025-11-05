

# **Proyecto Final — Predicción de precios de propiedades (Properati AR)**

## 📖 **Contexto**

Properati es un portal inmobiliario que reúne miles de publicaciones de propiedades en venta y alquiler en distintos países de Latinoamérica.  

El objetivo de este proyecto es construir un modelo de **Machine Learning** capaz de estimar el **precio de venta en dólares** de una propiedad en Argentina, a partir de sus características estructurales y de ubicación.

Este caso simula el trabajo de un equipo de Data Science que debe analizar [datos históricos de Properati](https://www.kaggle.com/datasets/alejandroczernikier/properati-argentina-dataset), definir un conjunto de variables relevantes, entrenar un modelo predictivo y finalmente **desplegar una aplicación interactiva** que permita estimar precios en tiempo real.

El desarrollo se divide en tres partes: **Análisis Exploratorio de Datos (EDA)**, **Modelado** y **Deploy**. Las partes de EDA y modelado se presentarán como un repositorio en GitHub (se evaluará cómo se trabajó con esta herramienta).

---

## 📊 **Parte 1 — EDA y Preparación de Datos**

### 🎯 **Objetivo**
Explorar el conjunto de datos, aplicar los filtros designados y preparar los datos para la fase de modelado.

### 🔹 **Pasos Sugeridos**
1.  **Cargar el dataset original de Properati**.
2.  **Filtrar** las propiedades según los siguientes criterios:
    *   `country_name = "Argentina"`
    *   `operation_type = "Venta"`
    *   `currency = "USD"`
    *   Zonas: **CABA** y **GBA** (pueden subdividir si lo desean).
3.  **Explorar variables relevantes**:
    *   Superficie (`surface_total`, `surface_covered`)
    *   Ambientes, dormitorios, baños
    *   Ubicación (`place_name`, `state_name`, `lat`, `lon`)
    *   Tipo de propiedad (`property_type`)
    *   Precio (`price_usd`)
4.  **Limpieza y tratamiento de datos**:
    *   Gestionar valores duplicados y nulos irrelevantes.
    *   Decidir **qué variables conservar** para el modelo.
    *   Imputar valores faltantes (si corresponde).
    *   Detectar y manejar **outliers** (ej. establecer límites razonables de superficie/precio).
5.  **Generar un dataset limpio final** y guardarlo como `data/processed.csv`.

> 💡 **Nota:** Se evaluará el criterio de limpieza y el razonamiento detrás de cada decisión. No es necesario que todos apliquen los mismos filtros.

### 🔹 **Entregables de esta parte**
*   **Conclusiones de negocio:** Presentar conclusiones que sirvan para conocer los datos desde una perspectiva de negocio.
*   **Storytelling:** Las conclusiones deben estar apoyadas en una narrativa o storytelling que guíe el análisis.
*   **Código y comentarios:** Mostrar el código utilizado para obtener la información, junto con comentarios que expliquen los pasos realizados.

---

## 🤖 **Parte 2 — Modelado y Evaluación**

### 🎯 **Objetivo**
Entrenar un modelo de regresión para predecir la variable `price_usd`.

### 🔹 **Lineamientos**
*   **Modelo:** Pueden elegir el que consideren más adecuado (ej. **Linear Regression, RandomForest, XGBoost, LightGBM**, etc.).
*   **División de datos:** Separar los datos en conjuntos de **entrenamiento (train) y prueba (test)**. El porcentaje de división es libre, pero debe estar justificado.
*   **Proceso iterativo:** Para llegar al modelo final, se deben construir varios modelos intermedios. Es crucial explicar las conclusiones parciales obtenidas en cada iteración y por qué se eligió el modelo final.

### 🔹 **Métricas de Evaluación**
Utilizar al menos una de las siguientes métricas:
*   **RMSE** (Root Mean Squared Error)
*   **MAE** (Mean Absolute Error)
*   **R²:** Pueden utilizarlo como guía para evaluar el ajuste del modelo.

> 💡 **Extra:** Comparar RMSE y MAE puede ayudar a entender cómo los outliers están afectando el rendimiento del modelo.

### 🔹 **Entregables de esta parte**
*   **Notebook de modelado:** Incluir el proceso de entrenamiento, las métricas obtenidas y un breve análisis de los resultados.
*   **Insights del modelo:** Presentar al menos **dos insights** clave que se hayan descubierto durante el proceso de creación del modelo.
*   **Justificación de variables:** Explicar qué variables se incluyeron en el modelo final y cuáles se descartaron, fundamentando la decisión.
*   **Exportación del modelo:** Guardar el modelo entrenado (`model.pkl`) y, si aplica, el preprocesador (`preprocess.pkl` o el pipeline completo).

---

## 🖥️ **Parte 3 — Interfaz con Gradio y Deploy en Hugging Face Spaces**

### 🎯 **Objetivo**
Implementar una interfaz de usuario simple para que se puedan ingresar los datos de una propiedad y obtener una predicción de precio del modelo entrenado.

### 🔹 **Requisitos Mínimos**
*   **Aplicación en Gradio:** Desarrollar la interfaz utilizando el modelo entrenado.
*   **Inputs:** La interfaz debe permitir ingresar los valores de las variables que utiliza el modelo.
*   **Output:** La aplicación debe mostrar la predicción del precio en USD.
*   **Diseño:** El diseño es libre. A continuación, tienen un ejemplo:

![Pantalla principal de la app](app-gradio.png) 

### 🔹 **Deploy**
1.  Subir el proyecto a **Hugging Face Spaces** (utilizando el tipo de aplicación "Gradio").
2.  Verificar que la aplicación funcione correctamente en línea.
3.  Incluir en el archivo `README.md` del repositorio:
    *   El link al Space de Hugging Face.
    *   Una captura de pantalla de la aplicación en funcionamiento.
    *   Un ejemplo de uso del endpoint que proporciona Gradio una vez desplegado.

### 🔹 **Entregables de esta parte**
*   El archivo **README.md** actualizado con la información mencionada anteriormente.

---

## ✅ **Requisitos Generales del Proyecto**

*   **Funcionalidad:** Todo el código entregado debe funcionar sin arrojar errores.
*   **Nota conceptual:** Se valorarán positivamente los siguientes aspectos:
    *   **Insights de negocio adicionales** que se identifiquen.
    *   **Aprendizajes técnicos** significativos que se hayan obtenido durante la realización del proyecto.