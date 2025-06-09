"""
Simple FastAPI Demand Forecasting API

Features:
- Select date range for forecasting
- Choose model type (Prophet, ML models)
- Get 7-day predictions
- No CSV upload required

Usage:
    uvicorn fastapi_forecasting_app:app --host 0.0.0.0 --port 8070
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

# Try to import the pipeline
try:
    from model_training_pipeline import AdvancedForecastingPipeline
except ImportError:
    # Fallback import name
    try:
        from advanced_forecasting_pipeline import AdvancedForecastingPipeline
    except ImportError:
        print("❌ Could not import forecasting pipeline")
        AdvancedForecastingPipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="🔮 Demand Forecasting API",
    description="Simple demand forecasting with date range and model selection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline = None
model_metadata = None

# Response models
class ForecastResult(BaseModel):
    sku_id: str
    location: str
    forecast_next_7_days: List[float]
    model_used: str
    model_type: str

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the forecasting pipeline."""
    global pipeline, model_metadata
    
    logger.info("🚀 Starting Demand Forecasting API...")
    
    try:
        if AdvancedForecastingPipeline is None:
            logger.error("❌ Pipeline class not available")
            return
            
        pipeline = AdvancedForecastingPipeline(model_save_dir='saved_models')
        
        if not os.path.exists('saved_models'):
            logger.warning("⚠️ No saved models found")
            return
        
        model_metadata = pipeline.load_model_metadata()
        logger.info(f"✅ Loaded models. Best model: {model_metadata['best_model']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        model_metadata = None

# ================== MAIN ENDPOINTS ==================

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint."""
    return APIResponse(
        success=True,
        message="🔮 Demand Forecasting API",
        data={
            "endpoints": {
                "/health": "Health check",
                "/models": "Available models",
                "/forecast": "Generate forecast",
                "/combinations": "Available SKU-locations"
            }
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check."""
    global model_metadata
    
    if model_metadata is None:
        return APIResponse(
            success=False,
            message="❌ Models not loaded",
            timestamp=datetime.now().isoformat()
        )
    
    return APIResponse(
        success=True,
        message="✅ API healthy",
        data={"best_model": model_metadata["best_model"]},
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=APIResponse)
async def get_available_models():
    """Get available models for selection."""
    global model_metadata
    
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    models = {}
    for name, info in model_metadata["models"].items():
        models[name] = {
            "type": info["type"],
            "mae": info["mae"],
            "r2": info["r2"]
        }
    
    return APIResponse(
        success=True,
        message="✅ Available models",
        data={
            "models": models,
            "best_model": model_metadata["best_model"],
            "model_types": {
                "ML": ["RandomForest", "XGBoost", "GradientBoosting", "Ridge"],
                "TimeSeries": ["Prophet", "ARIMA"]
            }
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/combinations", response_model=APIResponse)
async def get_combinations():
    """Get available SKU-location combinations."""
    
    # Standard combinations based on training data
    combinations = [
        "apple456_Delhi", "apple456_Mumbai",
        "banana789_Delhi", "banana789_Mumbai", 
        "mango123_Delhi", "mango123_Mumbai"
    ]
    
    skus = ["apple456", "banana789", "mango123"]
    locations = ["Delhi", "Mumbai"]
    
    return APIResponse(
        success=True,
        message="✅ Available combinations",
        data={
            "combinations": combinations,
            "skus": skus,
            "locations": locations
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/forecast", response_model=APIResponse)
async def generate_forecast(
    start_date: str = Query("2025-04-08", description="Start date (YYYY-MM-DD)"),
    model_name: str = Query("best", description="Model name or 'best' for best model"),
    sku_id: Optional[str] = Query("mango123", description="Specific SKU (optional)"),
    location: Optional[str] = Query("Mumbai", description="Specific location (optional)")
):
    """
    Generate 7-day forecast.
    
    Parameters:
    - start_date: Date to start forecasting from (YYYY-MM-DD)
    - model_name: Model to use ('best', 'Prophet', 'RandomForest', etc.)
    - sku_id: Optional specific SKU to forecast
    - location: Optional specific location to forecast
    """
    global pipeline, model_metadata
    
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Parse start date
        try:
            forecast_start = pd.to_datetime(start_date)
        except:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Determine which model to use
        if model_name == "best":
            selected_model = model_metadata["best_model"]
        else:
            if model_name not in model_metadata["models"]:
                available_models = list(model_metadata["models"].keys())
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{model_name}' not found. Available: {available_models}"
                )
            selected_model = model_name
        
        model_info = model_metadata["models"][selected_model]
        logger.info(f"🔮 Using model: {selected_model} (Type: {model_info['type']})")
        
        # Get combinations to forecast
        all_combinations = [
            "apple456_Delhi", "apple456_Mumbai",
            "banana789_Delhi", "banana789_Mumbai", 
            "mango123_Delhi", "mango123_Mumbai"
        ]
        
        # Filter combinations based on parameters
        filtered_combinations = []
        for combo in all_combinations:
            combo_sku, combo_location = combo.split('_')
            
            if sku_id and combo_sku != sku_id:
                continue
            if location and combo_location != location:
                continue
                
            filtered_combinations.append(combo)
        
        if not filtered_combinations:
            raise HTTPException(status_code=400, detail="No matching combinations found")
        
        # Generate forecasts
        forecasts = []
        
        if model_info["type"] == "ML":
            # ML Model prediction
            default_rolling = 50.0  # Default rolling average
            
            for combo in filtered_combinations:
                combo_sku, combo_location = combo.split('_')
                
                # Create 7 days of prediction data
                forecast_dates = [forecast_start + timedelta(days=i) for i in range(7)]
                prediction_data = pd.DataFrame({
                    'order_date': forecast_dates,
                    'sku_id': [combo_sku] * 7,
                    'location': [combo_location] * 7,
                    'rolling_7d': [default_rolling] * 7
                })
                
                # Make predictions
                predictions = pipeline.predict_with_ml_model(selected_model, prediction_data)
                
                forecasts.append(ForecastResult(
                    sku_id=combo_sku,
                    location=combo_location,
                    forecast_next_7_days=predictions.tolist(),
                    model_used=selected_model,
                    model_type=model_info["type"]
                ))
        
        elif model_info["type"] == "TimeSeries":
            # Time Series Model prediction
            prediction_data_list = []
            for combo in filtered_combinations:
                combo_sku, combo_location = combo.split('_')
                prediction_data_list.append({
                    'order_date': forecast_start,
                    'sku_id': combo_sku,
                    'location': combo_location
                })
            
            prediction_data = pd.DataFrame(prediction_data_list)
            
            if selected_model == "Prophet":
                predictions_dict = pipeline.predict_with_prophet_models(prediction_data, forecast_steps=7)
            elif selected_model == "ARIMA":
                predictions_dict = pipeline.predict_with_arima_models(prediction_data, forecast_steps=7)
            else:
                raise HTTPException(status_code=500, detail=f"Unknown time series model: {selected_model}")
            
            # Convert predictions to response format
            for ts_key, predictions in predictions_dict.items():
                combo_sku, combo_location = ts_key.split('_')
                
                forecasts.append(ForecastResult(
                    sku_id=combo_sku,
                    location=combo_location,
                    forecast_next_7_days=predictions.tolist(),
                    model_used=selected_model,
                    model_type=model_info["type"]
                ))
        
        if not forecasts:
            raise HTTPException(status_code=400, detail="No forecasts generated")
        
        logger.info(f"✅ Generated {len(forecasts)} forecasts")
        
        response_data = {
            "forecasts": [f.dict() for f in forecasts],
            "summary": {
                "total_forecasts": len(forecasts),
                "forecast_start_date": start_date,
                "forecast_days": 7,
                "model_used": selected_model,
                "model_type": model_info["type"],
                "model_performance": {
                    "mae": model_info["mae"],
                    "r2": model_info["r2"]
                }
            }
        }
        
        return APIResponse(
            success=True,
            message=f"✅ 7-day forecast generated using {selected_model}",
            data=response_data,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.post("/forecast", response_model=APIResponse)
async def generate_forecast_post(
    start_date: str,
    model_name: str = "best",
    sku_id: Optional[str] = None,
    location: Optional[str] = None
):
    """POST version of forecast endpoint."""
    return await generate_forecast(start_date, model_name, sku_id, location)

# ================== EXAMPLE ENDPOINTS ==================

@app.get("/example", response_model=APIResponse)
async def get_examples():
    """Get usage examples."""
    
    examples = {
        "basic_usage": [
            {
                "description": "Forecast all SKU-locations using best model",
                "url": "/forecast?start_date=2024-12-01"
            },
            {
                "description": "Forecast using Prophet model",
                "url": "/forecast?start_date=2024-12-01&model_name=Prophet"
            },
            {
                "description": "Forecast specific SKU",
                "url": "/forecast?start_date=2024-12-01&sku_id=mango123"
            },
            {
                "description": "Forecast specific location",
                "url": "/forecast?start_date=2024-12-01&location=Mumbai"
            },
            {
                "description": "Forecast specific SKU at specific location using RandomForest",
                "url": "/forecast?start_date=2024-12-01&model_name=RandomForest&sku_id=apple456&location=Delhi"
            }
        ],
        "curl_examples": [
            'curl "http://localhost:8070/forecast?start_date=2024-12-01"',
            'curl "http://localhost:8070/forecast?start_date=2024-12-01&model_name=Prophet"',
            'curl "http://localhost:8070/forecast?start_date=2024-12-01&sku_id=mango123&location=Mumbai"'
        ],
        "response_format": {
            "success": True,
            "message": "✅ 7-day forecast generated using RandomForest",
            "data": {
                "forecasts": [
                    {
                        "sku_id": "mango123",
                        "location": "Mumbai",
                        "forecast_next_7_days": [105, 110, 95, 100, 98, 102, 105],
                        "model_used": "RandomForest",
                        "model_type": "ML"
                    }
                ],
                "summary": {
                    "total_forecasts": 1,
                    "forecast_start_date": "2024-12-01",
                    "forecast_days": 7,
                    "model_used": "RandomForest"
                }
            }
        }
    }
    
    return APIResponse(
        success=True,
        message="📖 API Usage Examples",
        data=examples,
        timestamp=datetime.now().isoformat()
    )

# ================== ERROR HANDLERS ==================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "🔍 Endpoint not found",
            "available_endpoints": ["/health", "/models", "/forecast", "/combinations", "/example"],
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "❌ Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI Demand Forecasting Server...")
    print("📖 API Documentation: http://localhost:8070/docs")
    print("🔍 Health Check: http://localhost:8070/health")
    uvicorn.run(app, host="0.0.0.0", port=8070) 