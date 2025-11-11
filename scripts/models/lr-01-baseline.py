import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

data_encoded = pd.read_csv("data/03-encoded-properati.csv", sep=',', index_col=0)

# MLFlow: Variables a setear para cada corrida del experimento

mlflow.set_experiment(experiment_name="baseline-model-lr")

test_split_size = 0.3
test_split_rs = 42
model_rs = 42
na_values_processing = "drop" # False, "fill", "drop"
max_price = 390611
columns_to_drop = [
    # "lat",
    # "lon",
    "days_since_start",
    "days_since_end",
    # "surface_uncovered",
    # "available_publication",
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

mlflow.log_param("test_split_size",test_split_size)
mlflow.log_param("test_split_rs",test_split_rs)
mlflow.log_param("model_rs",model_rs)
mlflow.log_param("na_values_processing",na_values_processing)
mlflow.log_param("max_price",max_price)
mlflow.log_param("columns_to_drop",", ".join(columns_to_drop))

data_encoded = data_encoded.drop(columns=columns_to_drop)
data_encoded = data_encoded[data_encoded["price"] < max_price]

if na_values_processing == "fill":
    condition_for_filling = data_encoded["lat"].isna() | data_encoded["lon"].notna()
    lat_mean = data_encoded["lat"].mean()
    lon_mean = data_encoded["lon"].mean()
    
    data_encoded["lat"] = data_encoded["lat"].mask(
        condition_for_filling,
        lat_mean
    )
    data_encoded["lon"] = data_encoded["lon"].mask(
        condition_for_filling,
        lon_mean
    )
elif na_values_processing == "drop":
    data_encoded = data_encoded[data_encoded["lat"].notna()]
    print(data_encoded.shape)

x_data = data_encoded.drop('price', axis=1)
y_data = data_encoded['price']

x_data = x_data.values
y_data = y_data.values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_split_size, random_state=test_split_rs)

model_reg = LinearRegression()
model_reg.fit(x_train, y_train)

# Evaluación de métricas

y_train_pred = model_reg.predict(x_train)
y_test_pred = model_reg.predict(x_test)

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

# MLFlow model params y metrics
mlflow.log_metric("rmse_train",rmse_train)
mlflow.log_metric("mae_train",mae_train)
mlflow.log_metric("r2_train",r2_train)
mlflow.log_metric("rmse_test",rmse_test)
mlflow.log_metric("mae_test",mae_test)
mlflow.log_metric("r2_test",r2_test)

# MLFlow model
# #mlflow.sklearn.log_model(
#     sk_model=model_reg,
#     name="baseline-model-lr"
# )