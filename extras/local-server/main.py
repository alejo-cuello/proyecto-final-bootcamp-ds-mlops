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

MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "rf-final-model.pkl")
CATEGORIES_OHE_PATH = os.path.join(SCRIPT_DIR, "model", "categories-ohe.pkl")

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
    
    response = {"Price_in_USD": score}
    
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
