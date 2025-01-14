from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import io

app = FastAPI()

@app.post("/preprocess_with_file/")
async def preprocess_with_file(
    file: UploadFile,
    imputation_method: str = Form(..., description="Choose 'mean' or 'median' for missing data imputation"),
):
    """
    Preprocess a dataset using Feature Engine methods.
    - Automatically detects numeric and categorical columns.
    - Applies missing value imputation and one-hot encoding.
    """
    try:
        # Read the uploaded file
        file_content = await file.read()
        data = pd.read_csv(io.StringIO(file_content.decode("utf-8")))

        # Automatically detect numeric and categorical columns
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()

        # Create preprocessing pipeline
        pipeline = Pipeline(steps=[
            ("imputer", MeanMedianImputer(imputation_method=imputation_method, variables=numeric_columns)),
            ("encoder", OneHotEncoder(variables=categorical_columns)),
            ("scaler", StandardScaler()),
        ])

        # Apply preprocessing
        processed_data = pipeline.fit_transform(data)

        # Return the results as JSON
        return JSONResponse(content={
            "original_data": data.to_dict(orient="records"),
            "processed_data": processed_data.tolist(),
            "categorical_columns": categorical_columns,
            "numeric_columns": numeric_columns,
            "message": "Preprocessing completed using OneHotEncoder for categorical columns."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
