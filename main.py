from typing import Union
from fastapi import FastAPI
import pandas as pd
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

def create_preprocessing_pipeline():
    pipeline = Pipeline(steps=[
        ("imputer", MeanMedianImputer(imputation_method='mean', variables=['feature1', 'feature2'])),
        ("encoder", OneHotEncoder(variables=['categorical_feature'])),
        ("scaler", StandardScaler())
    ])
    return pipeline

app = FastAPI()

# for model and pipeline
model = None
pipeline = None

# Request schemas
class DataInput(BaseModel):
    data: list  # A list of dictionaries representing rows
    features: list  # Column names


@app.get("/")    #for API check.
def check():
    return {"status": "API is running"}


@app.post("/preprocess")
def preprocess(data: DataInput):
    global pipeline

    if not pipeline:
        pipeline = create_preprocessing_pipeline()
    
    try:
        df = pd.DataFrame(data.data, columns=data.features)
        processed_data = pipeline.fit_transform(df)
        return {"processed_data": processed_data.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/train")
def train(data: DataInput):
    from sklearn.ensemble import RandomForestClassifier

    global model, pipeline
    if not pipeline:
        pipeline = create_preprocessing_pipeline()

    try:
        df = pd.DataFrame(data.data, columns=data.features)
        X = pipeline.fit_transform(df.drop(columns=["target"]))
        y = df["target"]
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(data: DataInput):
    global model, pipeline

    if not model or not pipeline:
        raise HTTPException(status_code=400, detail="Model not trained or pipeline not defined")

    try:
        df = pd.DataFrame(data.data, columns=data.features)
        X = pipeline.transform(df)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))