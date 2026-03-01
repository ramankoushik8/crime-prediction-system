# Crime Prediction & Analysis System

## Overview
An end-to-end machine learning and data engineering pipeline designed to predict crime incident patterns and proactive resource allocation. Processing multi-format data (CSV, JSON, XML), this system evaluates multiple ML algorithms to achieve highly accurate predictions.

## Features
- **ETL Pipeline:** Robust data ingestion supporting CSV, JSON, and XML with median imputation for missing values.
- **Machine Learning:** Benchmarks Decision Tree, Gaussian Naïve Bayes, Linear SVM, and Linear/Quadratic Regression models.
- **Production Ready:** Exposes a real-time REST API using FastAPI for instant predictions.

## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd crime-prediction-system

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Train the Model:**
    Place your dataset in the `data/` folder, then run the training script:
    ```bash
    cd src
    python train.py
    
4. **Run the API:**
    ```bash
    uvicorn app:app --reload
    ```
    Navigate to `http://127.0.0.1:0000/docs` to test the API via the interactive Swagger UI.
