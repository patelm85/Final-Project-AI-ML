"""
FastAPI application for serving Walmart sales forecasting models.

This API loads three registered models from MLflow Model Registry and provides
endpoints for making predictions.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
# Import model flavors - these are dynamically loaded by mlflow
# We'll use mlflow.sklearn.load_model and mlflow.lightgbm.load_model directly

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="REST API for serving ML models trained on Walmart sales data",
    version="1.0.0"
)

# Global variables to store loaded models
model1 = None  # XGBoost
model2 = None  # LightGBM
model3 = None  # RandomForest


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    store: int = Field(..., description="Store number", ge=1)
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    holiday_flag: int = Field(..., description="Holiday flag (0 or 1)", ge=0, le=1)
    temperature: float = Field(..., description="Temperature")
    fuel_price: float = Field(..., description="Fuel price", ge=0)
    cpi: float = Field(..., description="Consumer Price Index", ge=0)
    unemployment: float = Field(..., description="Unemployment rate", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "store": 1,
                "date": "2012-11-01",
                "holiday_flag": 0,
                "temperature": 65.5,
                "fuel_price": 3.45,
                "cpi": 211.0,
                "unemployment": 7.5
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    data: List[PredictionRequest]


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: float = Field(..., description="Predicted weekly sales")
    model: str = Field(..., description="Model name used for prediction")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[float] = Field(..., description="List of predicted weekly sales")
    model: str = Field(..., description="Model name used for prediction")


def prepare_features(
    store: int,
    date: str,
    holiday_flag: int,
    temperature: float,
    fuel_price: float,
    cpi: float,
    unemployment: float
) -> pd.DataFrame:
    """
    Prepare features for prediction in the same format as training data.
    
    Args:
        store: Store number
        date: Date string in YYYY-MM-DD format
        holiday_flag: Holiday flag (0 or 1)
        temperature: Temperature
        fuel_price: Fuel price
        cpi: Consumer Price Index
        unemployment: Unemployment rate
        
    Returns:
        DataFrame with features ready for model prediction
    """
    # Parse date
    date_obj = pd.to_datetime(date)
    
    # Extract temporal features
    year = date_obj.year
    month = date_obj.month
    week = date_obj.isocalendar().week
    day_of_year = date_obj.dayofyear
    
    # Create feature DataFrame in the same order as training
    features = pd.DataFrame({
        'Store': [store],
        'Holiday_Flag': [holiday_flag],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Year': [year],
        'Month': [month],
        'Week': [week],
        'DayOfYear': [day_of_year]
    })
    
    return features


def load_model_from_latest_run(experiment_name: str, run_name_pattern: str, model_type: str = "sklearn"):
    """
    Load a model from the latest run in an experiment that matches a name pattern.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name_pattern: Pattern to match in run names (e.g., "xgboost", "lightgbm")
        model_type: Type of model flavor ("sklearn" or "lightgbm")
        
    Returns:
        Loaded model or None if not found
    """
    try:
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"  Experiment '{experiment_name}' not found")
            return None
        
        # Get all runs from the experiment, ordered by start time (most recent first)
        # We'll filter by run name pattern manually since MLflow filter syntax can be tricky
        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100  # Get enough runs to find the matching one
        )
        
        # Filter runs by name pattern (case-insensitive)
        matching_runs = [
            r for r in all_runs 
            if run_name_pattern.lower() in r.data.tags.get('mlflow.runName', '').lower()
        ]
        
        if not matching_runs:
            print(f"  No runs found matching pattern '{run_name_pattern}'")
            return None
        
        # Get the most recent matching run
        latest_run = matching_runs[0]
        
        run_id = latest_run.info.run_id
        run_name = latest_run.data.tags.get('mlflow.runName', run_id)
        print(f"  Found run: {run_name} (ID: {run_id})")
        
        # Load model from run
        if model_type == "lightgbm":
            model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        else:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        return model
    except Exception as e:
        print(f"  Error loading from latest run: {str(e)}")
        return None


def load_models():
    """Load all three models from MLflow Model Registry or latest runs."""
    global model1, model2, model3
    
    # Set MLflow tracking URI
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        print("WARNING: MLFLOW_TRACKING_URI environment variable not set")
        print("Models will not be loaded. Please set MLFLOW_TRACKING_URI before starting the API.")
        print("Example: export MLFLOW_TRACKING_URI='http://56.228.2.60:5000'")
        return
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "walmart-sales-forecast"
    
    # Load XGBoost model (Model 1)
    print("Loading XGBoost model...")
    try:
        # Try Model Registry first
        model1 = mlflow.sklearn.load_model("models:/XGBoost_Sales_Forecast/Production")
        print("XGBoost model loaded from Model Registry (Production)")
    except Exception as e1:
        try:
            # Try latest version in Model Registry
            model1 = mlflow.sklearn.load_model("models:/XGBoost_Sales_Forecast/latest")
            print("XGBoost model loaded from Model Registry (latest)")
        except Exception as e2:
            # Try loading from latest run in experiment
            print("  Model Registry not available, trying latest run...")
            model1 = load_model_from_latest_run(experiment_name, "xgboost", "sklearn")
            if model1:
                print("XGBoost model loaded from latest run")
            else:
                print("Could not load XGBoost model")
    
    # Load LightGBM model (Model 2)
    print("\nLoading LightGBM model...")
    try:
        # Try Model Registry first
        model2 = mlflow.lightgbm.load_model("models:/LightGBM_Sales_Forecast/Production")
        print("LightGBM model loaded from Model Registry (Production)")
    except Exception as e1:
        try:
            # Try latest version in Model Registry
            model2 = mlflow.lightgbm.load_model("models:/LightGBM_Sales_Forecast/latest")
            print("LightGBM model loaded from Model Registry (latest)")
        except Exception as e2:
            # Try loading from latest run in experiment
            print("  Model Registry not available, trying latest run...")
            model2 = load_model_from_latest_run(experiment_name, "lightgbm", "lightgbm")
            if model2:
                print("LightGBM model loaded from latest run")
            else:
                print("Could not load LightGBM model")
    
    # Load RandomForest model (Model 3)
    print("\nLoading RandomForest model...")
    try:
        # Try Model Registry first
        model3 = mlflow.sklearn.load_model("models:/RandomForest_Sales_Forecast/Production")
        print("RandomForest model loaded from Model Registry (Production)")
    except Exception as e1:
        try:
            # Try latest version in Model Registry
            model3 = mlflow.sklearn.load_model("models:/RandomForest_Sales_Forecast/latest")
            print("RandomForest model loaded from Model Registry (latest)")
        except Exception as e2:
            # Try loading from latest run in experiment
            print("  Model Registry not available, trying latest run...")
            model3 = load_model_from_latest_run(experiment_name, "random", "sklearn")
            if model3:
                print("RandomForest model loaded from latest run")
            else:
                print("Could not load RandomForest model")


@app.on_event("startup")
async def startup_event():
    """Load models when the API starts."""
    print("="*70)
    print("Starting Walmart Sales Forecasting API...")
    print("="*70)
    load_models()
    print("="*70)
    print("API ready to serve predictions!")
    print("="*70)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_status = {
        "model1_xgboost": model1 is not None,
        "model2_lightgbm": model2 is not None,
        "model3_randomforest": model3 is not None
    }
    
    all_loaded = all(models_status.values())
    
    return {
        "status": "healthy" if all_loaded else "degraded",
        "models": models_status,
        "message": "All models loaded" if all_loaded else "Some models failed to load"
    }


@app.post("/predict_model1", response_model=PredictionResponse)
async def predict_model1(request: PredictionRequest):
    """
    Make a prediction using Model 1 (XGBoost).
    
    Args:
        request: Prediction request with input features
        
    Returns:
        Prediction response with predicted weekly sales
    """
    if model1 is None:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model is not loaded. Check /health endpoint for model status."
        )
    
    try:
        # Prepare features
        features = prepare_features(
            store=request.store,
            date=request.date,
            holiday_flag=request.holiday_flag,
            temperature=request.temperature,
            fuel_price=request.fuel_price,
            cpi=request.cpi,
            unemployment=request.unemployment
        )
        
        # Make prediction
        prediction = model1.predict(features)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model="XGBoost_Sales_Forecast"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_model2", response_model=PredictionResponse)
async def predict_model2(request: PredictionRequest):
    """
    Make a prediction using Model 2 (LightGBM).
    
    Args:
        request: Prediction request with input features
        
    Returns:
        Prediction response with predicted weekly sales
    """
    if model2 is None:
        raise HTTPException(
            status_code=503,
            detail="LightGBM model is not loaded. Check /health endpoint for model status."
        )
    
    try:
        # Prepare features
        features = prepare_features(
            store=request.store,
            date=request.date,
            holiday_flag=request.holiday_flag,
            temperature=request.temperature,
            fuel_price=request.fuel_price,
            cpi=request.cpi,
            unemployment=request.unemployment
        )
        
        # Make prediction
        prediction = model2.predict(features)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model="LightGBM_Sales_Forecast"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_model3", response_model=PredictionResponse)
async def predict_model3(request: PredictionRequest):
    """
    Make a prediction using Model 3 (RandomForest).
    
    Args:
        request: Prediction request with input features
        
    Returns:
        Prediction response with predicted weekly sales
    """
    if model3 is None:
        raise HTTPException(
            status_code=503,
            detail="RandomForest model is not loaded. Check /health endpoint for model status."
        )
    
    try:
        # Prepare features
        features = prepare_features(
            store=request.store,
            date=request.date,
            holiday_flag=request.holiday_flag,
            temperature=request.temperature,
            fuel_price=request.fuel_price,
            cpi=request.cpi,
            unemployment=request.unemployment
        )
        
        # Make prediction
        prediction = model3.predict(features)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model="RandomForest_Sales_Forecast"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_model1/batch", response_model=BatchPredictionResponse)
async def predict_model1_batch(request: BatchPredictionRequest):
    """Make batch predictions using Model 1 (XGBoost)."""
    if model1 is None:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model is not loaded. Check /health endpoint for model status."
        )
    
    try:
        # Prepare features for all requests
        features_list = []
        for req in request.data:
            features = prepare_features(
                store=req.store,
                date=req.date,
                holiday_flag=req.holiday_flag,
                temperature=req.temperature,
                fuel_price=req.fuel_price,
                cpi=req.cpi,
                unemployment=req.unemployment
            )
            features_list.append(features)
        
        # Concatenate all features
        all_features = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        predictions = model1.predict(all_features).tolist()
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model="XGBoost_Sales_Forecast"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_model2/batch", response_model=BatchPredictionResponse)
async def predict_model2_batch(request: BatchPredictionRequest):
    """Make batch predictions using Model 2 (LightGBM)."""
    if model2 is None:
        raise HTTPException(
            status_code=503,
            detail="LightGBM model is not loaded. Check /health endpoint for model status."
        )
    
    try:
        # Prepare features for all requests
        features_list = []
        for req in request.data:
            features = prepare_features(
                store=req.store,
                date=req.date,
                holiday_flag=req.holiday_flag,
                temperature=req.temperature,
                fuel_price=req.fuel_price,
                cpi=req.cpi,
                unemployment=req.unemployment
            )
            features_list.append(features)
        
        # Concatenate all features
        all_features = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        predictions = model2.predict(all_features).tolist()
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model="LightGBM_Sales_Forecast"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_model3/batch", response_model=BatchPredictionResponse)
async def predict_model3_batch(request: BatchPredictionRequest):
    """Make batch predictions using Model 3 (RandomForest)."""
    if model3 is None:
        raise HTTPException(
            status_code=503,
            detail="RandomForest model is not loaded. Check /health endpoint for model status."
        )
    
    try:
        # Prepare features for all requests
        features_list = []
        for req in request.data:
            features = prepare_features(
                store=req.store,
                date=req.date,
                holiday_flag=req.holiday_flag,
                temperature=req.temperature,
                fuel_price=req.fuel_price,
                cpi=req.cpi,
                unemployment=req.unemployment
            )
            features_list.append(features)
        
        # Concatenate all features
        all_features = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        predictions = model3.predict(all_features).tolist()
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model="RandomForest_Sales_Forecast"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

