import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor

class KNNMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_neighbors=5):
        self.cols = cols
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = self.imputer.transform(X_new[self.cols])
        return X_new

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

class IterativeMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, max_iter=10):
        self.cols = cols
        self.max_iter = max_iter
        self.imputer = IterativeImputer(max_iter=self.max_iter)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = self.imputer.transform(X_new[self.cols])
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class CategoricalMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, strategy='most_frequent', fill_value='missing'):
        self.cols = cols
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = self.imputer.transform(X_new[self.cols])
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.cols:
            X_new[col + '_missing_ind'] = X_new[col].isnull().astype(int)
        return X_new

    def fit_transform(self, X, y=None):
        return self.transform(X)

class GroupBasedMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col, agg_func='median'):
        self.group_col = group_col
        self.target_col = target_col
        self.agg_func = agg_func
        self.group_map = None

    def fit(self, X, y=None):
        self.group_map = X.groupby(self.group_col)[self.target_col].agg(self.agg_func)
        return self

    def transform(self, X):
        X_new = X.copy()
        mask = X_new[self.target_col].isnull()
        X_new.loc[mask, self.target_col] = X_new.loc[mask, self.group_col].map(self.group_map)
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class RandomForestAgeImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing Age values using RandomForestRegressor based on other features.
    Rounds ages and sets any zero values to 1.
    """
    def __init__(self, features=None, age_col='Age'):
        self.features = features if features is not None else ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
        self.age_col = age_col
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y=None):
        df = X.copy()
        # Encode Sex if present
        if 'Sex' in self.features and 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        mask = df[self.age_col].notna()
        self.model.fit(df.loc[mask, self.features], df.loc[mask, self.age_col])
        return self

    def transform(self, X):
        df = X.copy()
        # Encode Sex if present
        if 'Sex' in self.features and 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        mask = df[self.age_col].isna()
        if mask.any():
            pred_ages = self.model.predict(df.loc[mask, self.features])
            # Round and convert to pandas nullable Int64
            df.loc[mask, self.age_col] = pd.Series(np.round(pred_ages), index=df.loc[mask].index).astype('Int64')
        # Round all ages and set zero to 1
        df[self.age_col] = df[self.age_col].round().astype('Int64')
        df.loc[df[self.age_col] == 0, self.age_col] = 1
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
