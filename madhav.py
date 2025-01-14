from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Optional
import pandas as pd
from feature_engine.imputation import MeanMedianImputer
import shutil
app = FastAPI()

@app.post("/upload-dataset/")
async def upload_and_impute(file: UploadFile = File(...),
imputation_method: Optional[str] = "median",
):
    try:
        if imputation_method not in ["mean", "median"]:
            raise HTTPException(
                status_code=400, detail="Invalid imputation method. Choose 'mean' or 'median'."
            )
        file_path = f"./{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        df = pd.read_csv(file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")
        imputer = MeanMedianImputer(imputation_method=imputation_method)
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