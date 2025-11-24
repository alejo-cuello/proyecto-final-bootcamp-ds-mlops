import os
import pickle
import pandas as pd
import uvicorn

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()

CURRENT_FILE = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(CURRENT_FILE)
ONE_LEVEL_UP = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(ONE_LEVEL_UP)

MODEL_PATH = os.path.join(ROOT_DIR, "mlruns","512582443179615027","models","m-0248ec91bbc349f393da1c30e4f3fed1","artifacts","model.pkl")
CATEGORIES_OHE_PATH = os.path.join(ROOT_DIR, "notebooks", "categories_ohe.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(CATEGORIES_OHE_PATH, "rb") as handle:
    columns_ohe = pickle.load(handle)

class Answer(BaseModel):
    rooms: int
    bedrooms: int
    bathrooms: int
    surface_total: float
    surface_covered: float
    l2: str
    property_type: str
    lat: float
    lon: float

@app.get("/")
async def root():
    return {"message": "Proyecto final del Bootcamp de DS y MLOps"}


@app.post("/predict")
def predict_price(answer: Answer):

    answer_dict = jsonable_encoder(answer)
    
    for key, value in answer_dict.items():
        answer_dict[key] = [value]
    
    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    single_instance_ohe = pd.get_dummies(single_instance, dtype="int64").reindex(columns=columns_ohe,fill_value=0)
    
    prediction = model.predict(single_instance_ohe)
 
    score = round(prediction[0],2)
    
    response = {"score": score}
    
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
