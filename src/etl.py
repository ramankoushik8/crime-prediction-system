import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

class CrimeDataETL:
    # Changed default target to index 127 based on the raw UCI dataset
    def __init__(self, target_column=127, threshold=0.1):
        self.target_column = target_column
        self.threshold = threshold
        self.imputer = SimpleImputer(strategy='median')

    def load_data(self, file_path):
        """Dynamically loads data based on file extension."""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.csv':
            # Added header=None because the raw UCI dataset has no column names
            df = pd.read_csv(file_path, header=None, na_values='?')
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.xml':
            df = pd.read_xml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return df

    def transform(self, df):
        """Cleans data and engineers the target variable."""
        # Drop non-predictive string columns (first 5 columns: indices 0 to 4)
        cols_to_drop = [0, 1, 2, 3, 4]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Create the binary target variable based on the original notebook logic
        if self.target_column in df.columns:
            df['highCrime'] = (df[self.target_column] > self.threshold).astype(int)
            # Drop the original continuous target to prevent data leakage
            df = df.drop(columns=[self.target_column])
        
        # Separate features and target
        if 'highCrime' in df.columns:
            X = df.drop(columns=['highCrime'])
            y = df['highCrime']
        else:
            X = df
            y = None

        # Impute missing values (replacing '?' with median instead of hardcoded 0)
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        return X_imputed, y
