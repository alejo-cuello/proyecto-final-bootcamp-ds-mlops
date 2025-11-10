import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

data_encoded = pd.read_csv("data/03-encoded-properati.csv", sep=',', index_col=0)

# MLFlow: Variables a setear para cada corrida del experimento

mlflow.set_experiment(experiment_name="baseline-model-rfr")

sample_frac = 0.2
sample_rs = 42 # rs = random_state
test_split_size = 0.3
test_split_rs = 42
model_rs = 42
random_grid_n_iter = 10
random_grid_cv = 3
random_grid_rs = 42
columns_to_drop = [
    "lat",
    "lon",
    "surface_uncovered",
    "available_publication",
    # "rooms",
    # "bedrooms",
    # "bathrooms",
    # "surface_total",
    # "surface_covered",
    # "price",
    # "days_since_start",
    # "days_since_end",
    # "l2_Bs.As. G.B.A. Zona Oeste",
    # "l2_Bs.As. G.B.A. Zona Sur",
    # "l2_Capital Federal",
    # "property_type_Departamento",
    # "property_type_Local comercial",
    # "property_type_Oficina",
    # "property_type_PH"
]

mlflow.log_param("sample_frac",sample_frac)
mlflow.log_param("sample_rs",sample_rs)
mlflow.log_param("test_split_size",test_split_size)
mlflow.log_param("test_split_rs",test_split_rs)
mlflow.log_param("model_rs",model_rs)
mlflow.log_param("random_grid_n_iter",random_grid_n_iter)
mlflow.log_param("random_grid_cv",random_grid_cv)
mlflow.log_param("random_grid_rs",random_grid_rs)
mlflow.log_param("columns_to_drop",", ".join(columns_to_drop))

# Aclaración importante: tomaré solo el 20% del total de registros para entrenar los primeros modelos que me permitirán evaluar y seleccionar los mejores modelos y sus features correspondientes. El 20% es un porcentaje arbitrario que creo que es bastante representativo del total, y son bastantes registros.

sample = data_encoded.sample(frac=sample_frac, random_state=sample_rs)
sample = sample.drop(columns=columns_to_drop)

categoric_cols = sample.select_dtypes(["object","bool"]).columns
for col in categoric_cols:
    sample[col] = sample[col].astype('category')

x_data = sample.drop('price', axis=1)
y_data = sample['price']

x_data = x_data.values
y_data = y_data.values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_split_size, random_state=test_split_rs)

model_rfr = RandomForestRegressor(random_state=model_rs)

# Realizaré una randomized search acotada debido a que tengo muchos registros. Primero intenté hacer esto con todos los registros y con 100 iteraciones, pero resultó inviable el tiempo que estaba tardando en entrenar

random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 5)],
    'max_features': [1.0],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [False] # Al tener muchos registros, creo que el remuestreo es innecesario
}

rf_random = RandomizedSearchCV(estimator = model_rfr, param_distributions = random_grid, scoring="neg_mean_absolute_error", n_iter = random_grid_n_iter, cv = random_grid_cv, verbose=2, random_state=random_grid_rs, n_jobs = -1)

rf_random.fit(x_train, y_train)

# Evaluación de métricas

y_train_pred = rf_random.predict(x_train)
y_test_pred = rf_random.predict(x_test)

def regression_metrics(y_true, y_test_pred):
    mse  = mean_squared_error(y_true, y_test_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_test_pred)
    r2   = r2_score(y_true, y_test_pred)
    
    return rmse, mae, r2

rmse_train, mae_train, r2_train = regression_metrics(y_train, y_train_pred)
print("Medidas en train:")
print("RMSE:", round(rmse_train, 2))
print("MAE:", round(mae_train, 2))
print("R²:", round(r2_train, 4))

rmse_test, mae_test, r2_test = regression_metrics(y_test, y_test_pred)
print("")
print("Medidas en test:")
print("RMSE:", round(rmse_test, 2))
print("MAE:", round(mae_test, 2))
print("R²:", round(r2_test, 4))

# MLFlow metrics
mlflow.log_metric("rmse_train",rmse_train)
mlflow.log_metric("mae_train",mae_train)
mlflow.log_metric("r2_train",r2_train)
mlflow.log_metric("rmse_test",rmse_test)
mlflow.log_metric("mae_test",mae_test)
mlflow.log_metric("r2_test",r2_test)

# MLFlow model
mlflow.sklearn.log_model(
    sk_model=rf_random,
    name="baseline-model-rfr"
)

# Dejo código comentado que podría servirme más adelante

# param_grid = {
#     'bootstrap': [True], # Base: True
#     'max_depth': [40, 50, 60, 70, 80], # Base: 60
#     'max_features': [1.0], # Base: 1.0
#     'min_samples_leaf': [1, 2, 3], # Base: 1
#     'min_samples_split': [2, 3, 4], # Base: 2
#     'n_estimators': [700, 750, 800] # Base: 733
# }

# grid_search = GridSearchCV(estimator = model_rfr, param_grid = param_grid, scoring = 'neg_mean_absolute_error', cv = 3, n_jobs = -1, verbose = 2)

# grid_search.fit(x_train, y_train)

# print("RandomizerSearch: Mejor combinación")
# print("Score en Train: " + str(rf_random.score(x_train, y_train)))
# print("Score en Test: " + str(rf_random.score(x_test, y_test)))
# print()
# print("GridSearch: Mejor combinación")
# print("Score en Train: " + str(grid_search.score(x_train, y_train)))
# print("Score en Test: " + str(grid_search.score(x_test, y_test)))

# vis_pred_err = PredictionError(rf_random)

# vis_pred_err.fit(x_train, y_train)  # Fiteamos los datos al visualizador
# vis_pred_err.score(x_test, y_test)  # Calculamos las métricas para test
# vis_pred_err.show()                 # Visualizamos!

# vis_res = ResidualsPlot(rf_random)

# # Copy-paste de la doc oficial: 
# vis_res.fit(x_train, y_train)  # Fiteamos los datos al visualizador
# vis_res.score(x_test, y_test)  # Calculamos las métricas para test

# plt.xticks(rotation=90)                # rotamos etiquetas eje x

# vis_res.show()                 # Visualizamos!


