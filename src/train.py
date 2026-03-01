import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from etl import CrimeDataETL

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=42),
            "Gaussian Naive Bayes": GaussianNB(),
            "Linear SVM": LinearSVC(random_state=42, max_iter=10000, dual='auto')
        }
        self.best_model = None
        self.best_name = None

    def evaluate_classification(self, y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"--- {model_name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")
        return f1

    def train_and_benchmark(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        best_f1 = 0
        
        # Train Classification Models
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = self.evaluate_classification(y_test, y_pred, name)
            
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_name = name
                
        print(f"Best Model: {self.best_name} (F1: {best_f1:.4f})")
        
        # Save the best model and the ETL imputer for production
        joblib.dump(self.best_model, '../models/best_model.pkl')
        print("Model saved to models/best_model.pkl")

if __name__ == "__main__":
    print("Loading data and starting ETL pipeline...")
    etl = CrimeDataETL()
    
    # Load the data (returns a single dataframe)
    df = etl.load_data('../data/communities-crime-full.csv')
    
    # Transform and clean the data (this is where it splits into X and y)
    X_transformed, y_transformed = etl.transform(df)
    
    print("Data processed successfully. Starting model training...\n")
    
    # Train and benchmark the models
    trainer = ModelTrainer()
    trainer.train_and_benchmark(X_transformed, y_transformed)
