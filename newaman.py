from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from feature_engine.imputation import MeanMedianImputer
import shutil

app = FastAPI()

class ImputationRequest(BaseModel):
    imputation_method: Optional[str] = "median"
    columns: Optional[List[str]] = None  # List of columns to apply imputation

@app.get("/")  # for API check.
def check():
    return {"status": "API is running"}

@app.post("/upload-dataset/")
async def upload_and_impute(
    request: ImputationRequest,  # Required parameter comes first
    file: UploadFile = File(...),  # Optional parameter with a default value
):
    try:
        # Debugging: Print request body
        print("Received Request:", request)

        imputation_method = request.imputation_method
        columns = request.columns

        if imputation_method not in ["mean", "median"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid imputation method. Choose 'mean' or 'median'.",
            )

        file_path = f"./{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")

        # Validate columns
        if columns:
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"The following columns are not in the dataset: {missing_columns}",
                )
            # Filter only the specified columns for imputation
            columns_to_impute = columns
        else:
            # If no columns specified, use all numeric columns by default
            columns_to_impute = df.select_dtypes(include=["number"]).columns.tolist()

        if not columns_to_impute:
            raise HTTPException(
                status_code=400,
                detail="No valid columns found for imputation.",
            )

        # Apply imputer to selected columns
        imputer = MeanMedianImputer(
            imputation_method=imputation_method, variables=columns_to_impute
        )
        imputer.fit(df)
        transformed_df = imputer.transform(df)

        transformed_file_path = f"./transformed_{file.filename}"
        transformed_df.to_csv(transformed_file_path, index=False)

        return {
            "message": "Missing values imputed successfully!",
            "imputed_file": transformed_file_path,
            "imputer_details": imputer.imputer_dict_,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
