from __future__ import print_function

# Author: Yanmichel Morfa morfayanmichel@gmail.com
# modified from a sklearn example by Gael Varoquaux

# License: BSD 3 clause
import os
import numpy as np
import pandas as pd


from sklearn import metrics
from sklearn import covariance, manifold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#local imports
import utils.maps_utils as mu
from utils.utils import loadmat, savemat
from utils.plotting_tools import plt, plot_embedding
from interpolation.gridding import interpolate

print(__doc__)


# =========================================================================
# get observations series and model predictions
path = 'data/train_data'
var_key = 't2m'
domain = 'd01'
init = '00'


data_path = os.path.join(path, '_'.join([var_key,  domain, init, 'unique']) )
data_dict = loadmat(data_path, squeeze_me=True)

#
for key in ['__header__', '__version__', '__globals__']:
    _ = data_dict.pop(key)

names = []
values = []
stn_id = []
for stn, value in data_dict.iteritems():

    stn_id.append(stn)
    names.append(stn.split('_')[-1])

    # computing mean squared error for stations
    x = value[:, 2]
    y = value[:, 1]

    metric = np.square(y - x)

    print('Fetching station: {}, with error variance: {}'.format(stn, metric.var()) )
    values.append( metric.squeeze() )

# =========================================================================
# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
stn_id = np.array(stn_id)
X = np.stack(values, axis=-1)
X /= X.std(axis=0)

print(np.shape(X))

# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV(n_refinements=10, cv=5)
edge_model.fit(X)

covariance = edge_model.covariance_
partial_correlations = edge_model.precision_

print(27 * '=', 'covariance matrix', 27 * '=')
print(covariance)
print(73 * '=')
# =========================================================================
# Cluster using affinity propagation / Kmeans methods

# unknown number of clusters:
estimator = AffinityPropagation(verbose=True)

# known number of clusters:
# estimator = KMeans(init='k-means++', n_clusters=5, n_init=100,
#                     precompute_distances=True, n_jobs=-1)

labels = estimator.fit_predict(covariance)

n_labels = labels.max()
print('{} different labels were found...'.format(n_labels+1))
stn_clusters = {}
for i in range(n_labels + 1):

    c_name = 'cluster_%i' % (i + 1)
    s_name = stn_id[labels == i]
    stn_clusters[c_name] = s_name

    print(c_name + ': ', ', '.join(s_name))

# save clusters id:
savemat('data/clustered_stations_' + var_key, stn_clusters)

print(62 * '=')
print('homogeneity\tcompleteness\tv-measure\tARI\tAMI')
print('%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t%.3f' % (
      metrics.homogeneity_score(labels, estimator.labels_),
      metrics.completeness_score(labels, estimator.labels_),
      metrics.v_measure_score(labels, estimator.labels_),
      metrics.adjusted_rand_score(labels, estimator.labels_),
      metrics.adjusted_mutual_info_score(labels,  estimator.labels_)))
print(62 * '=')

# =========================================================================
# Find a low-dimension embedding for visualization: find the best position
# of the nodes (the stations) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=12)

embedding = node_position_model.fit_transform(X.T).T

figure = plot_embedding(partial_correlations, embedding, names, labels)
figure.savefig('outputs/stations_embedding_' + var_key + '.png')

#============= geographic visualization of the clustering =================

stn_id, stn_lon, stn_lat = np.loadtxt('data/estaciones.dt').T

stn_id = [ str(stn) for stn in stn_id.astype('int32') ]

indices = []
colors = np.empty_like(stn_id); colors.fill(np.nan)
for name, label in zip(names, labels):

    if name in stn_id:
        index = stn_id.index(name)
        colors[index] = label
        indices.append(index)

# Create map object
proj_dict = {'rsphere': 6370997.0, 'lat_0': 22.0, 'lon_0': -80.0,
             'lat_1': 15.0, 'lat_2': 35.0, 'proj': 'lcc', 'res': 'h'}
mapa = mu.get_mapa(stn_lon, stn_lat, proj=proj_dict, buff=0.2)

figure = plt.figure(figsize=(8., 3.2), dpi=300)
mapa.drawcoastlines(linewidth=0.3)

mapa.scatter(stn_lon, stn_lat,
    marker='o', s=15, edgecolors='k', linewidth=0.7,
    c=colors, latlon=True)

for ind in indices:

    color = colors[ind]
    (x, y) = mapa(stn_lon[ind], stn_lat[ind])
    plt.text(x, y, stn_id[ind], size=3.5, bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(int(color) / float(n_labels)),
                       alpha=.5))

figure.savefig('outputs/plots/station_clusters_' + var_key + '.png')
mu.plt.close(figure)
