import pandas as pd

class MissingValueHandler:
    def handle(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ["object", "category"]:
                    X[col] = X[col].fillna("Unknown")
                else:
                    X[col] = X[col].fillna(X[col].median())
        return X
