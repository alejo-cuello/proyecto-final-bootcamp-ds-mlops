import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

data_encoded = pd.read_csv("data/03-encoded-properati.csv", sep=',', index_col=0)

# MLFlow: Variables a setear para cada corrida del experimento

mlflow.set_experiment(experiment_name="gridsearchcv-model-rfr")

sample_frac = 0.2
sample_rs = 42 # rs = random_state
test_split_size = 0.2
test_split_rs = 42
model_rs = 42
random_grid_cv = 3
max_price = 390611
columns_to_drop = [
    # "lat",
    # "lon",
    "surface_uncovered",
    "available_publication",
    "days_since_start",
    "days_since_end",
    # "rooms",
    # "bedrooms",
    # "bathrooms",
    # "surface_total",
    # "surface_covered",
    # "price",
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
mlflow.log_param("random_grid_cv",random_grid_cv)
mlflow.log_param("max_price",max_price)
mlflow.log_param("columns_to_drop",", ".join(columns_to_drop))

# Aclaración importante: tomaré solo el 20% del total de registros para entrenar los primeros modelos que me permitirán evaluar y seleccionar los mejores modelos y sus features correspondientes. El 20% es un porcentaje arbitrario que creo que es bastante representativo del total, y son bastantes registros.

sample = data_encoded.sample(frac=sample_frac, random_state=sample_rs)
sample = sample[sample["price"] < max_price]
sample = sample.drop(columns=columns_to_drop)

x_data = sample.drop('price', axis=1)
y_data = sample['price']

x_data = x_data.values
y_data = y_data.values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_split_size, random_state=test_split_rs)

model_rfr = RandomForestRegressor(random_state=model_rs)

# Realizaré una gridSearch tomando como eje los parámetros obtenidos en el mejor run del experimento baseline-model-rfr
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [10, 15],
#     'max_features': [1.0],
#     'min_samples_leaf': [2, 3, 4],
#     'min_samples_split': [5, 7, 10],
#     'n_estimators': [200]
# }

# Prueba extra para ver si se reduce el overfitting
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [8, 10, 12],
#     'max_features': [1.0],
#     'min_samples_leaf': [2,3],
#     'min_samples_split': [6,7,8],
#     'n_estimators': [200]
# }

# Parámetros del modelo final
param_grid = {
    'bootstrap': [False],
    'max_depth': [12],
    'max_features': [1.0],
    'min_samples_leaf': [3],
    'min_samples_split': [8],
    'n_estimators': [200]
}

grid_search = GridSearchCV(estimator = model_rfr, param_grid = param_grid, scoring = "neg_mean_absolute_error", cv = random_grid_cv, verbose=2, n_jobs = -1)
grid_search.fit(x_train, y_train)

# Evaluación de métricas

y_train_pred = grid_search.predict(x_train)
y_test_pred = grid_search.predict(x_test)

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

for param in grid_search.best_params_:
    mlflow.log_param("param_"+param,grid_search.best_params_[param])

# MLFlow model params y metrics
mlflow.log_metric("rmse_train",rmse_train)
mlflow.log_metric("mae_train",mae_train)
mlflow.log_metric("r2_train",r2_train)
mlflow.log_metric("rmse_test",rmse_test)
mlflow.log_metric("mae_test",mae_test)
mlflow.log_metric("r2_test",r2_test)

# MLFlow model
signature = infer_signature(x_train, grid_search.predict(x_train))
columns_to_drop.append("price")
columns = data_encoded.drop(columns=columns_to_drop).columns
x_train_df = pd.DataFrame(x_train, columns=columns)
input_example = x_train_df.head(5)

mlflow.sklearn.log_model(
    sk_model=grid_search,
    name="rfr",
    signature=signature,
    input_example=input_example,
    registered_model_name="rfr"
)