import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, RobustScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly = None
        self.feature_names = None

    def fit(self, X, y=None):
        self.poly = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=self.include_bias)
        self.poly.fit(X)
        self.feature_names = self.poly.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        X_poly = self.poly.transform(X)
        return pd.DataFrame(X_poly, columns=self.feature_names, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class TargetEncoder(BaseEstimator, TransformerMixin):
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

class FrequencyEncoder(BaseEstimator, TransformerMixin):
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

class Binner(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_bins=5, strategy='quantile'):
        self.cols = cols
        self.n_bins = n_bins
        self.strategy = strategy
        self.binners = {}

    def fit(self, X, y=None):
        for col in self.cols:
            kb = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
            kb.fit(X[[col]])
            self.binners[col] = kb
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.cols:
            X_new[col + '_bin'] = self.binners[col].transform(X_new[[col]]).astype(int)
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class GroupAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, agg_col, agg_func='mean'):
        self.group_col = group_col
        self.agg_col = agg_col
        self.agg_func = agg_func
        self.agg_map = None

    def fit(self, X, y=None):
        self.agg_map = X.groupby(self.group_col)[self.agg_col].agg(self.agg_func)
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[f'{self.agg_col}_grp_{self.agg_func}'] = X_new[self.group_col].map(self.agg_map)
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = self.scaler.transform(X_new[self.cols])
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class QuantileScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, output_distribution='uniform'):
        self.cols = cols
        self.scaler = QuantileTransformer(output_distribution=output_distribution)

    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = self.scaler.transform(X_new[self.cols])
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selected_features = None

    def fit(self, X, y=None):
        self.selector.fit(X)
        self.selected_features = X.columns[self.selector.get_support()]
        return self

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.selected_features = None

    def fit(self, X, y):
        mi = mutual_info_classif(X, y)
        idx = np.argsort(mi)[-self.n_features:]
        self.selected_features = X.columns[idx]
        return self

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

