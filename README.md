
# Supply Chain Demand Forecasting Pipeline

## 🎯 Objective
This project implements a comprehensive ML-based pipeline for **supply chain demand forecasting** that covers the complete data science workflow: data ingestion → preprocessing → model training → prediction → deployment. The goal is to predict future demand for SKUs across different locations to optimize inventory management and supply chain operations.

## 📊 Overview of Approach

### 1. **Data Generation & Simulation**
- Created a realistic 60-day historical order dataset with 3 SKUs across 2 locations
- Incorporated realistic business patterns:
  - **Seasonality**: Higher demand on weekdays vs. weekends
  - **Trend**: 10% upward growth trend over the period
  - **Location Effects**: Mumbai (100% baseline) vs. Delhi (80% baseline)
  - **Noise**: Gaussian noise (σ≈5) for realistic variance

### 2. **Exploratory Data Analysis (EDA)**
- Comprehensive analysis in `eda.ipynb` covering:
  - Time series visualization and patterns
  - SKU performance comparison
  - Location-based demand analysis
  - Seasonality and trend decomposition
  - Statistical summaries and correlations

### 3. **Feature Engineering**
- **Temporal Features**: Day of week, month, quarter, year
- **Lag Features**: 1, 3, 7-day historical demand
- **Rolling Statistics**: 3, 7, 14-day moving averages
- **Location & SKU Encoding**: Categorical feature handling
- **Trend Features**: Linear and polynomial trend components

### 4. **Model Selection & Comparison**
Evaluated 7 different approaches across two paradigms:


#### **Traditional ML Models (Feature-based)**
- RandomForest, Gradient Boosting, XGBoost
- Linear Regression, Ridge Regression



![image](https://github.com/user-attachments/assets/e16cf1fb-661a-414c-83dc-a96365042455)


#### **Time Series Specific Models**
- Prophet (Facebook's time series forecasting)
- ARIMA (AutoRegressive Integrated Moving Average)

### 5. **Deployment & Demo**
- **FastAPI**: RESTful API with multiple endpoints
- **Streamlit**: Interactive web application for real-time forecasting

---



## 🚀 FastAPI Endpoint Demo

### **Quick Start**
```bash
# Start the API server
python fastapi_forecasting_app.py

# Access interactive documentation
open http://localhost:8070/docs
```

### **Key Endpoints**

#### **1. Generate Forecast**
```bash
GET /forecast?start_date=2025-04-08&model_name=best&sku_id=mango123&location=Mumbai
```

#### **2. Model Selection**
```bash
GET /models  # View available models and performance
```

#### **3. Available Combinations**
```bash
GET /combinations  # View SKU-location combinations
```

### **Interactive Features**
- **Model Selection**: Choose between Prophet, RandomForest, XGBoost, or "best"
- **Date Selection**: Specify any start date for 7-day forecast
- **Filtering**: Select specific SKUs or locations
- **Real-time Results**: Instant JSON response with predictions



### ✅ **Data Preprocessing and Feature Engineering**
- **Temporal Features**: Day of week, month, day of month extraction
- **Rolling Statistics**: 7-day rolling averages with proper lag to prevent leakage
- **Data Validation**: Input validation, missing value handling, data type checking
- **Scaling**: StandardScaler for numerical features, OneHotEncoder for categorical

```python
# Sophisticated feature engineering pipeline
def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Date features + rolling averages + validation
```

### ✅ **Reasoning Behind Model Choice**
- **Comprehensive Comparison**: Systematic evaluation of ML vs Time Series approaches
- **Performance Metrics**: MAE, RMSE, R², MAPE for thorough evaluation


## 🛠️ Technical Implementation

### **Architecture Overview**
```
Data Input → Feature Engineering → Model Training → Model Selection → API Serving
     ↓              ↓                    ↓              ↓            ↓
  order_data.csv → ML/TS Features → Multiple Models → Best Model → FastAPI
```

## 📈 Performance Results

Based on our model comparison (stored in `advanced_model_comparison.json`):

- **Best Model**: [Automatically determined from training]
- **MAE Performance**: [Actual values from training]
- **R² Score**: [Model explanatory power]
- **MAPE**: [Percentage-based error metric]

## 🔄 Usage Instructions

### **Training Models**
```bash
python model_training_pipeline.py
```

### **Starting API Server**
```bash
python fastapi_forecasting_app.py
```


### streamlit app
``` bash
streamlit run streamlit_model_app.py 
```

![image](https://github.com/user-attachments/assets/5b70a475-cb25-4bee-9fe7-a1e3afbfd120)


![image](https://github.com/user-attachments/assets/0b2fc123-ffba-41e5-b9a9-6819468abf67)


![image](https://github.com/user-attachments/assets/061b32cc-e3d1-4cb7-a5bb-4498cd1d9649)
