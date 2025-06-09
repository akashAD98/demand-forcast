"""
Advanced Demand Forecasting Pipeline with Time Series Models

This module compares traditional ML models with specialized time series models:
- ARIMA (AutoRegressive Integrated Moving Average)
- Prophet (Facebook's time series forecasting)
- Traditional ML models (RandomForest, XGBoost, etc.)

Author: Demand Forecasting Team
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, List
warnings.filterwarnings('ignore')

# Traditional ML Libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Time Series Libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    print("⚠️ statsmodels not available. ARIMA models will be skipped.")
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️ Prophet not available. Prophet models will be skipped.")
    PROPHET_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Continuing with other models.")
    XGBOOST_AVAILABLE = False

class AdvancedForecastingPipeline:
    """
    Advanced pipeline comparing ML models with time series models.
    
    Model Categories:
    1. Traditional ML: Uses features (SKU, location, day_of_week, etc.)
    2. Time Series: Uses only time and quantity (ARIMA, Prophet)
    3. Hybrid: Combines both approaches
    """
    
    def __init__(self, data_path: str = 'order_data.csv', model_save_dir: str = 'saved_models'):
        self.data_path = data_path
        self.model_save_dir = model_save_dir
        self.target = 'quantity'
        self.ml_features = ['sku_id', 'location', 'day_of_week', 'day_of_month', 'month', 'rolling_7d']
        self.cat_features = ['sku_id', 'location', 'day_of_week']
        self.num_features = ['day_of_month', 'month', 'rolling_7d']
        self.results = {}
        
        # Create model save directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load and parse the order data."""
        print("📊 Loading data...")
        df = pd.read_csv(self.data_path, parse_dates=['order_date'])
        print(f"✅ Data loaded: {df.shape[0]} records from {df['order_date'].min()} to {df['order_date'].max()}")
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for traditional ML models."""
        print("🔧 Engineering ML features...")
        df = df.copy()
        
        # Date-based features
        df['day_of_week'] = df['order_date'].dt.weekday
        df['day_of_month'] = df['order_date'].dt.day
        df['month'] = df['order_date'].dt.month
        
        # Sort for proper rolling calculation
        df = df.sort_values(['sku_id', 'location', 'order_date'])
        
        # Rolling average (lagged to prevent data leakage)
        df['rolling_7d'] = df.groupby(['sku_id', 'location'])['quantity'].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        
        # Drop rows with NaN rolling averages
        df = df.dropna(subset=['rolling_7d'])
        
        print(f"✅ ML features ready: {df.shape[0]} records")
        return df
    
    def prepare_time_series_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare data for time series models (separate series per SKU-location)."""
        print("📅 Preparing time series data...")
        
        time_series_dict = {}
        
        # Create separate time series for each SKU-location combination
        for (sku, location), group in df.groupby(['sku_id', 'location']):
            ts_key = f"{sku}_{location}"
            
            # Sort by date and create time series
            ts_data = group.sort_values('order_date')[['order_date', 'quantity']].copy()
            ts_data = ts_data.set_index('order_date').asfreq('D', method='ffill')
            
            time_series_dict[ts_key] = ts_data
        
        print(f"✅ Created {len(time_series_dict)} time series")
        return time_series_dict
    
    def split_data(self, df: pd.DataFrame, holdout_days: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based train/test split."""
        print(f"📊 Splitting data with {holdout_days} days holdout...")
        max_date = df['order_date'].max()
        cutoff_date = max_date - timedelta(days=holdout_days)
        
        train_df = df[df['order_date'] <= cutoff_date]
        test_df = df[df['order_date'] > cutoff_date]
        
        print(f"✅ Train: {train_df.shape[0]} records | Test: {test_df.shape[0]} records")
        return train_df, test_df
    
    # ==================== ARIMA MODELS ====================
    
    def fit_arima_model(self, ts_data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Any:
        """Fit ARIMA model to a single time series."""
        try:
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            print(f"   ⚠️ ARIMA fitting failed: {str(e)}")
            return None
    
    def auto_arima_order(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """Simple auto ARIMA order selection (simplified version)."""
        # Check stationarity
        try:
            adf_test = adfuller(ts_data.dropna())
            is_stationary = adf_test[1] < 0.05
            d = 0 if is_stationary else 1
            
            # Try different p and q values
            best_aic = float('inf')
            best_order = (1, d, 1)
            
            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
            
            return best_order
        except:
            return (1, 1, 1)  # Default order
    
    def train_arima_models(self, time_series_dict: Dict[str, pd.DataFrame], 
                          train_end_date: pd.Timestamp) -> Dict[str, Any]:
        """Train ARIMA models for each time series."""
        print("\n🔥 Training ARIMA models...")
        
        if not ARIMA_AVAILABLE:
            print("❌ ARIMA not available")
            return {}
        
        arima_models = {}
        
        for ts_key, ts_data in time_series_dict.items():
            print(f"   Training ARIMA for {ts_key}...")
            
            # Split into train/test
            train_ts = ts_data[ts_data.index <= train_end_date]['quantity']
            
            if len(train_ts) < 10:  # Need minimum data
                print(f"   ⚠️ Insufficient data for {ts_key}")
                continue
            
            # Auto-select ARIMA order
            order = self.auto_arima_order(train_ts)
            print(f"   Selected ARIMA order: {order}")
            
            # Fit model
            model = self.fit_arima_model(train_ts, order)
            if model is not None:
                arima_models[ts_key] = model
                print(f"   ✅ ARIMA fitted for {ts_key}")
        
        print(f"✅ ARIMA models trained for {len(arima_models)} series")
        return arima_models
    
    # ==================== PROPHET MODELS ====================
    
    def fit_prophet_model(self, ts_data: pd.DataFrame) -> Any:
        """Fit Prophet model to a single time series."""
        try:
            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            prophet_data = pd.DataFrame({
                'ds': ts_data.index,
                'y': ts_data['quantity'].values
            })
            
            # Initialize and fit Prophet
            model = Prophet(
                yearly_seasonality=False,  # Only 2 months of data
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            
            model.fit(prophet_data)
            return model
        except Exception as e:
            print(f"   ⚠️ Prophet fitting failed: {str(e)}")
            return None
    
    def train_prophet_models(self, time_series_dict: Dict[str, pd.DataFrame], 
                           train_end_date: pd.Timestamp) -> Dict[str, Any]:
        """Train Prophet models for each time series."""
        print("\n🔥 Training Prophet models...")
        
        if not PROPHET_AVAILABLE:
            print("❌ Prophet not available")
            return {}
        
        prophet_models = {}
        
        for ts_key, ts_data in time_series_dict.items():
            print(f"   Training Prophet for {ts_key}...")
            
            # Split into train/test
            train_ts = ts_data[ts_data.index <= train_end_date]
            
            if len(train_ts) < 14:  # Prophet needs minimum data
                print(f"   ⚠️ Insufficient data for {ts_key}")
                continue
            
            # Fit model
            model = self.fit_prophet_model(train_ts)
            if model is not None:
                prophet_models[ts_key] = model
                print(f"   ✅ Prophet fitted for {ts_key}")
        
        print(f"✅ Prophet models trained for {len(prophet_models)} series")
        return prophet_models
    
    # ==================== TRADITIONAL ML MODELS ====================
    
    def build_preprocessor(self):
        """Build preprocessing pipeline for ML models."""
        return ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_features),
            ('num', StandardScaler(), self.num_features)
        ], remainder='drop')
    
    def train_ml_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Train traditional ML models."""
        print("\n🔥 Training ML models...")
        
        X_train = train_df[self.ml_features]
        y_train = train_df[self.target]
        X_test = test_df[self.ml_features]
        y_test = test_df[self.target]
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        
        ml_results = {}
        preprocessor = self.build_preprocessor()
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            ml_results[name] = {
                'model': pipeline,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'type': 'ML'
            }
            
            print(f"   ✅ {name}: MAE={mae:.2f}, R²={r2:.3f}")
        
        return ml_results
    
    # ==================== EVALUATION ====================
    
    def evaluate_time_series_models(self, models_dict: Dict[str, Any], 
                                   time_series_dict: Dict[str, pd.DataFrame],
                                   test_start_date: pd.Timestamp,
                                   model_type: str) -> Dict[str, Dict]:
        """Evaluate time series models (ARIMA/Prophet)."""
        print(f"\n📊 Evaluating {model_type} models...")
        
        results = {}
        all_predictions = []
        all_actuals = []
        
        for ts_key, model in models_dict.items():
            if ts_key not in time_series_dict:
                continue
                
            ts_data = time_series_dict[ts_key]
            test_ts = ts_data[ts_data.index > test_start_date]['quantity']
            
            if len(test_ts) == 0:
                continue
            
            try:
                if model_type == 'ARIMA':
                    # ARIMA forecast
                    forecast = model.forecast(steps=len(test_ts))
                    # Handle different ARIMA output formats
                    if hasattr(forecast, 'values'):
                        predictions = forecast.values
                    elif isinstance(forecast, np.ndarray):
                        predictions = forecast
                    elif isinstance(forecast, (list, tuple)):
                        predictions = np.array(forecast)
                    else:
                        predictions = np.array([forecast])
                
                elif model_type == 'Prophet':
                    # Prophet forecast
                    future_dates = pd.DataFrame({'ds': test_ts.index})
                    forecast = model.predict(future_dates)
                    predictions = forecast['yhat'].values
                
                # Convert to numpy arrays and ensure same length
                predictions = np.array(predictions).flatten()
                actuals = np.array(test_ts.values).flatten()
                
                min_len = min(len(predictions), len(actuals))
                if min_len > 0:
                    predictions = predictions[:min_len]
                    actuals = actuals[:min_len]
                    
                    all_predictions.extend(predictions)
                    all_actuals.extend(actuals)
                
            except Exception as e:
                print(f"   ⚠️ Error evaluating {ts_key}: {str(e)}")
                continue
        
        if len(all_predictions) > 0:
            # Calculate overall metrics
            mae = mean_absolute_error(all_actuals, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
            r2 = r2_score(all_actuals, all_predictions)
            mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100
            
            results[model_type] = {
                'models': models_dict,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'type': 'TimeSeries',
                'num_series': len(models_dict)
            }
            
            print(f"   ✅ {model_type}: MAE={mae:.2f}, R²={r2:.3f} ({len(models_dict)} series)")
        
        return results
    
    def compare_all_models(self, all_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """Compare all model types and select the best."""
        print("\n🏆 Model Comparison Results:")
        print("=" * 80)
        
        # Sort by MAE
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mae'])
        
        for i, (name, result) in enumerate(sorted_results):
            rank = i + 1
            model_type = result.get('type', 'Unknown')
            
            if model_type == 'TimeSeries':
                extra_info = f"({result['num_series']} series)"
            else:
                extra_info = "(feature-based)"
            
            print(f"{rank}. {name:15} | MAE: {result['mae']:6.2f} | R²: {result['r2']:6.3f} | "
                  f"MAPE: {result['mape']:5.1f}% | {extra_info}")
        
        best_model_name, best_result = sorted_results[0]
        
        print("=" * 80)
        print(f"🥇 BEST MODEL: {best_model_name}")
        print(f"   Type: {best_result.get('type', 'Unknown')}")
        print(f"   MAE: {best_result['mae']:.2f}")
        print(f"   R²: {best_result['r2']:.3f}")
        print(f"   MAPE: {best_result['mape']:.1f}%")
        
        return best_model_name, best_result
    
    # ==================== MODEL SAVING & LOADING ====================
    
    def save_ml_models(self, ml_results: Dict[str, Any]) -> None:
        """Save ML models using joblib (sklearn pipelines)."""
        print("\n💾 Saving ML models...")
        
        for model_name, result in ml_results.items():
            if result.get('type') == 'ML':
                # ML models are sklearn pipelines - save with joblib
                model_path = os.path.join(self.model_save_dir, f"{model_name}.joblib")
                joblib.dump(result['model'], model_path)
                print(f"   ✅ {model_name} saved to {model_path}")
    
    def save_prophet_models(self, prophet_results: Dict[str, Dict]) -> None:
        """Save Prophet models (each time series separately)."""
        print("\n💾 Saving Prophet models...")
        
        for model_type, result in prophet_results.items():
            if model_type == 'Prophet' and 'models' in result:
                # Create directory for Prophet models
                prophet_dir = os.path.join(self.model_save_dir, 'Prophet_models')
                os.makedirs(prophet_dir, exist_ok=True)
                
                # Save each Prophet model (per time series)
                for ts_key, prophet_model in result['models'].items():
                    # Prophet models can be saved with joblib too
                    model_path = os.path.join(prophet_dir, f"{ts_key}.joblib")
                    joblib.dump(prophet_model, model_path)
                    print(f"   ✅ Prophet model for {ts_key} saved")
                
                print(f"   📁 All Prophet models saved in {prophet_dir}")
    
    def save_arima_models(self, arima_results: Dict[str, Dict]) -> None:
        """Save ARIMA models (each time series separately)."""
        print("\n💾 Saving ARIMA models...")
        
        for model_type, result in arima_results.items():
            if model_type == 'ARIMA' and 'models' in result:
                # Create directory for ARIMA models
                arima_dir = os.path.join(self.model_save_dir, 'ARIMA_models')
                os.makedirs(arima_dir, exist_ok=True)
                
                # Save each ARIMA model (per time series)
                for ts_key, arima_model in result['models'].items():
                    # ARIMA models from statsmodels can be saved with joblib
                    model_path = os.path.join(arima_dir, f"{ts_key}.joblib")
                    joblib.dump(arima_model, model_path)
                    print(f"   ✅ ARIMA model for {ts_key} saved")
                
                print(f"   📁 All ARIMA models saved in {arima_dir}")
    
    def save_model_metadata(self, all_results: Dict[str, Dict], best_model_name: str) -> None:
        """Save model performance metadata."""
        print("\n💾 Saving model metadata...")
        
        metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'models': {}
        }
        
        for model_name, result in all_results.items():
            metadata['models'][model_name] = {
                'type': result.get('type', 'Unknown'),
                'mae': float(result['mae']),
                'rmse': float(result['rmse']),
                'r2': float(result['r2']),
                'mape': float(result['mape']),
                'num_series': result.get('num_series', 1)
            }
        
        metadata_path = os.path.join(self.model_save_dir, 'models_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Metadata saved to {metadata_path}")
    
    def load_ml_model(self, model_name: str):
        """Load a specific ML model."""
        model_path = os.path.join(self.model_save_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"ML model {model_name} not found at {model_path}")
    
    def load_prophet_models(self) -> Dict[str, Any]:
        """Load all Prophet models."""
        prophet_dir = os.path.join(self.model_save_dir, 'Prophet_models')
        prophet_models = {}
        
        if os.path.exists(prophet_dir):
            for file in os.listdir(prophet_dir):
                if file.endswith('.joblib'):
                    ts_key = file.replace('.joblib', '')
                    model_path = os.path.join(prophet_dir, file)
                    prophet_models[ts_key] = joblib.load(model_path)
        
        return prophet_models
    
    def load_arima_models(self) -> Dict[str, Any]:
        """Load all ARIMA models."""
        arima_dir = os.path.join(self.model_save_dir, 'ARIMA_models')
        arima_models = {}
        
        if os.path.exists(arima_dir):
            for file in os.listdir(arima_dir):
                if file.endswith('.joblib'):
                    ts_key = file.replace('.joblib', '')
                    model_path = os.path.join(arima_dir, file)
                    arima_models[ts_key] = joblib.load(model_path)
        
        return arima_models
    
    def load_model_metadata(self) -> Dict:
        """Load model performance metadata."""
        metadata_path = os.path.join(self.model_save_dir, 'models_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    # ==================== INFERENCE METHODS ====================
    
    def predict_with_ml_model(self, model_name: str, prediction_data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a saved ML model."""
        print(f"🔮 Loading and predicting with {model_name}...")
        
        # Load the ML model
        model = self.load_ml_model(model_name)
        
        # Prepare features (same as training)
        prediction_data = self.prepare_prediction_features(prediction_data)
        X_pred = prediction_data[self.ml_features]
        
        # Make predictions
        predictions = model.predict(X_pred)
        print(f"✅ Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_with_prophet_models(self, prediction_data: pd.DataFrame, forecast_steps: int = 7) -> Dict[str, np.ndarray]:
        """Make predictions using saved Prophet models."""
        print(f"🔮 Loading and predicting with Prophet models...")
        
        # Load all Prophet models
        prophet_models = self.load_prophet_models()
        predictions = {}
        
        # Get unique SKU-location combinations from prediction data
        for (sku, location), group in prediction_data.groupby(['sku_id', 'location']):
            ts_key = f"{sku}_{location}"
            
            if ts_key in prophet_models:
                prophet_model = prophet_models[ts_key]
                
                # Create future dates for Prophet
                start_date = group['order_date'].min()
                future_dates = pd.date_range(start=start_date, periods=forecast_steps, freq='D')
                future_df = pd.DataFrame({'ds': future_dates})
                
                # Make prediction
                forecast = prophet_model.predict(future_df)
                predictions[ts_key] = forecast['yhat'].values
                
                print(f"   ✅ Predicted for {ts_key}")
        
        print(f"✅ Generated predictions for {len(predictions)} time series")
        return predictions
    
    def predict_with_arima_models(self, prediction_data: pd.DataFrame, forecast_steps: int = 7) -> Dict[str, np.ndarray]:
        """Make predictions using saved ARIMA models."""
        print(f"🔮 Loading and predicting with ARIMA models...")
        
        # Load all ARIMA models
        arima_models = self.load_arima_models()
        predictions = {}
        
        # Get unique SKU-location combinations from prediction data
        for (sku, location), group in prediction_data.groupby(['sku_id', 'location']):
            ts_key = f"{sku}_{location}"
            
            if ts_key in arima_models:
                arima_model = arima_models[ts_key]
                
                # Make prediction
                forecast = arima_model.forecast(steps=forecast_steps)
                if hasattr(forecast, 'values'):
                    predictions[ts_key] = forecast.values
                else:
                    predictions[ts_key] = np.array(forecast)
                
                print(f"   ✅ Predicted for {ts_key}")
        
        print(f"✅ Generated predictions for {len(predictions)} time series")
        return predictions
    
    def prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction (same as training)."""
        df = df.copy()
        
        # Date-based features
        df['day_of_week'] = df['order_date'].dt.weekday
        df['day_of_month'] = df['order_date'].dt.day
        df['month'] = df['order_date'].dt.month
        
        # For rolling features, use provided value or default
        if 'rolling_7d' not in df.columns:
            print("⚠️ rolling_7d not provided, using default value 50.0")
            df['rolling_7d'] = 50.0
        
        return df
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete advanced forecasting pipeline."""
        print("🚀 Advanced Forecasting Pipeline - Comparing ML vs Time Series Models")
        print("=" * 80)
        
        # Load and prepare data
        df = self.load_data()
        
        # Prepare for ML models
        ml_df = self.prepare_ml_features(df)
        train_df, test_df = self.split_data(ml_df)
        
        # Prepare for time series models
        time_series_dict = self.prepare_time_series_data(df)
        train_end_date = train_df['order_date'].max()
        test_start_date = test_df['order_date'].min()
        
        all_results = {}
        
        # Train ML models
        ml_results = self.train_ml_models(train_df, test_df)
        all_results.update(ml_results)
        
        # Train ARIMA models
        if ARIMA_AVAILABLE:
            arima_models = self.train_arima_models(time_series_dict, train_end_date)
            if arima_models:
                arima_results = self.evaluate_time_series_models(
                    arima_models, time_series_dict, test_start_date, 'ARIMA'
                )
                all_results.update(arima_results)
        
        # Train Prophet models
        if PROPHET_AVAILABLE:
            prophet_models = self.train_prophet_models(time_series_dict, train_end_date)
            if prophet_models:
                prophet_results = self.evaluate_time_series_models(
                    prophet_models, time_series_dict, test_start_date, 'Prophet'
                )
                all_results.update(prophet_results)
        
        # Compare all models
        best_model_name, best_result = self.compare_all_models(all_results)
        
        # ==================== SAVE ALL TRAINED MODELS ====================
        print("\n💾 SAVING ALL TRAINED MODELS...")
        
        # Save ML models (RandomForest, XGBoost, etc.)
        self.save_ml_models(ml_results)
        
        # Save Prophet models (if trained)
        if 'Prophet' in all_results:
            self.save_prophet_models(all_results)
        
        # Save ARIMA models (if trained)
        if 'ARIMA' in all_results:
            self.save_arima_models(all_results)
        
        # Save model metadata
        self.save_model_metadata(all_results, best_model_name)
        
        # Save results summary
        results_metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_model_type': best_result.get('type', 'Unknown'),
            'best_mae': float(best_result['mae']),
            'best_r2': float(best_result['r2']),
            'best_mape': float(best_result['mape']),
            'models_compared': list(all_results.keys()),
            'arima_available': ARIMA_AVAILABLE,
            'prophet_available': PROPHET_AVAILABLE,
            'xgboost_available': XGBOOST_AVAILABLE,
            'models_saved_to': self.model_save_dir
        }
        
        with open('advanced_model_comparison.json', 'w') as f:
            json.dump(results_metadata, f, indent=2)
        
        print("\n🎉 Advanced pipeline completed!")
        print(f"📁 Results saved to: advanced_model_comparison.json")
        print(f"💾 ALL MODELS SAVED TO: {self.model_save_dir}")
        
        return results_metadata

def main():
    """Main execution function."""
    pipeline = AdvancedForecastingPipeline()
    results = pipeline.run_complete_pipeline()
    return results

if __name__ == "__main__":   
    main() 