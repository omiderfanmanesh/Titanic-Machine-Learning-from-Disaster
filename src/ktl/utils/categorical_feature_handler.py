import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class OneHotCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, drop='first', sparse=False):
        self.cols = cols
        self.drop = drop
        self.sparse = sparse
        self.encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse)
        self.feature_names = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        self.feature_names = self.encoder.get_feature_names_out(self.cols)
        return self

    def transform(self, X):
        X_new = X.copy()
        encoded = self.encoder.transform(X_new[self.cols])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names, index=X_new.index)
        X_new = X_new.drop(columns=self.cols)
        X_new = pd.concat([X_new, encoded_df], axis=1)
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class OrdinalCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = self.encoder.transform(X_new[self.cols])
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class TargetCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.target_means = {}

    def fit(self, X, y):
        for col in self.cols:
            self.target_means[col] = X.groupby(col)[y.name].mean()
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.cols:
            X_new[col + '_target_enc'] = X_new[col].map(self.target_means[col])
        return X_new

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class FrequencyCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in self.cols:
            self.freq_maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.cols:
            X_new[col + '_freq_enc'] = X_new[col].map(self.freq_maps[col])
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class LeaveOneOutCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.target_means = {}

    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in self.cols:
            self.target_means[col] = X.groupby(col)[y.name].mean()
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        for col in self.cols:
            if y is not None:
                means = X_new[col].map(self.target_means[col])
                # Leave-one-out: subtract the current row's target value
                X_new[col + '_loo_enc'] = (means * (X_new.groupby(col)[col].transform('count') - 1) - y) / (X_new.groupby(col)[col].transform('count') - 1)
                X_new[col + '_loo_enc'] = X_new[col + '_loo_enc'].fillna(self.global_mean)
            else:
                X_new[col + '_loo_enc'] = X_new[col].map(self.target_means[col]).fillna(self.global_mean)
        return X_new

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, min_freq=0.01, rare_label='Rare'):
        self.cols = cols
        self.min_freq = min_freq
        self.rare_label = rare_label
        self.rare_maps = {}

    def fit(self, X, y=None):
        for col in self.cols:
            freq = X[col].value_counts(normalize=True)
            rare_cats = freq[freq < self.min_freq].index
            self.rare_maps[col] = rare_cats
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.cols:
            X_new[col] = X_new[col].apply(lambda x: self.rare_label if x in self.rare_maps[col] else x)
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

