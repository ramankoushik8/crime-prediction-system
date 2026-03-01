# Crime Prediction & Analysis System

## Overview
An end-to-end data engineering and machine learning pipeline designed to predict crime incident patterns, enabling proactive law enforcement resource allocation. The system ingests and transforms real-world socio-economic and law enforcement data, trains multiple machine learning models to identify high-risk areas, and serves the winning model via a real-time REST API.

## Architecture & Features
- **Automated Data Sourcing:** Built-in script to fetch the 1990 US Census and FBI UCR data directly from the UCI Machine Learning Repository.
- **Robust ETL Pipeline:** Dynamically handles data ingestion, drops non-predictive string identifiers, engineers a binary `highCrime` target variable, and imputes missing values using statistical medians to prevent data leakage.
- **ML Benchmarking Pipeline:** Automatically trains and evaluates Decision Tree, Gaussian Naïve Bayes, and Linear SVC models, saving the highest-performing model (`best_model.pkl`) for production.
- **Real-Time Production API:** Exposes the trained model via a high-performance FastAPI server for instant inference.
- **Live Traffic Simulation:** Includes a terminal dashboard script that streams simulated police dispatch data to the API to verify real-time prediction accuracy and latency.

## Tech Stack
- **Language:** Python 3.12+
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, Joblib
- **API & Serving:** FastAPI, Uvicorn, Pydantic, Requests

---

## Project Structure
```text
crime-prediction-system/
│
├── data/                   # Raw datasets (Ignored in Git)
├── models/                 # Saved ML models (Ignored in Git)
├── src/
│   ├── __init__.py
│   ├── etl.py              # Data extraction and transformation class
│   ├── train.py            # Model training and benchmarking script
│   ├── app.py              # FastAPI production server
│   ├── test_api.py         # Single-request API test script
│   └── simulate_traffic.py # Live data streaming dashboard
│
├── .gitignore              # Git exclusion rules
├── download_data.py        # Automated UCI dataset downloader
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation & Setup

1. **Clone the repository:**
    ```bash
    git clone git@github.com:ramankoushik8/crime-prediction-system.git
    cd crime-prediction-system
2. **Install dependencies:**
    It is recommended to use a virtual environment or `pyenv`.
    ```bash
    pip install -r requirements.txt
3. **Download the Dataset:**
    Pull the Communities and Crime dataset into the `data/` folder:
    ```bash
    python download_data.py

## Usage
1. Train the Machine Learning Models
    Run the training pipeline to evaluate the algorithms and generate the `best_model.pkl` file.
    ```bash
    cd src
    python train.py
2. Start the Production API
    Launch the FastAPI server to serve the model.
    ```bash
    # Ensure you are still in the src/ directory
    uvicorn app:app --reload
    ```
    You can view the interactive API documentation by navigating to http://127.0.0.1:8000/docs in your browser.

3. Simulate Live Traffic
Open a second terminal window, navigate to the `src/` directory, and run the simulator to watch the API handle real-time streaming data:
    ```bash
    python simulate_traffic.py
