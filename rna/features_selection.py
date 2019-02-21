# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_models import GridSearchCV, models, rng, param_grid

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import RobustScaler
from utils.utils import loadmat
#
data = loadmat('data/train_data/t2m_rain_mslp_rh2m_wind_d01_00.mat')
data = data['stn_359']

X, y = data[:, 2:], data[:, 1]

print(X[0], y[0])

# scale the data first:
scaler = RobustScaler()

# This dataset is way too high-dimensional. Better do PCA:
# Maybe some original features where good, too?
pca = PCA()
selection = SelectKBest(k='all')

# Build estimator from PCA and Univariate selection:
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)

# Do grid search over k, n_components and C:
pipeline = Pipeline([("features", combined_features), ("scaler", scaler),
                     ("model", models['mlp_r'])])

features_range = range(1, X.shape[-1] + 1)

param_grid = dict(
    features__pca__n_components=features_range,
    features__univ_select__k=features_range,
    model__hidden_layer_sizes=[(10, 3), (3, ), (100, 150, 100, ),])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)

print('Best estimator: \n', grid_search.best_estimator_)
print('Best score:', grid_search.best_score_)
