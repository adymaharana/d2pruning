# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract class for sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import abc
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm
import sys

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None


# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""


class kCenterGreedy(SamplingMethod):

  def __init__(self, X, y, seed, metric='euclidean'):
    self.X = X
    self.y = y
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    if len(self.features.shape) == 1:
      self.features = self.features.reshape(1, -1)
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      if len(x.shape) == 1:
        x = x.reshape(1, -1)
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    # try:
    #   # Assumes that the transform function takes in original data and not
    #   # flattened data.
    #   print('Getting transformed features...')
    #   self.features = model.transform(self.X)
    #   print('Calculating distances...')
    #   self.update_distances(already_selected, only_new=False, reset_dist=True)
    # except:
    #   print('Using flat_X as features.')
    #   self.update_distances(already_selected, only_new=True, reset_dist=False)

    if N == 0:
      print("Skipping sampling because of 0 budget")
      return []

    new_batch = []
    print("Selecting %s-centers from %s pool" % (N, self.n_obs))
    for _ in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
        # New examples should not be in already selected since those points
        # should have min_distance of zero to a cluster center.
      if self.already_selected:
        assert ind not in self.already_selected, (self.already_selected, ind, self.min_distances)

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      self.already_selected = new_batch
    print('Maximum distance from cluster centers is %0.2f' % max(self.min_distances), '; selected %s centers' % len(new_batch))
    # self.already_selected = already_selected
    return new_batch


"""Diversity promoting sampling method that uses graph density to determine
 most representative points.

This is an implementation of the method described in
https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf
"""

class GraphDensitySampler(SamplingMethod):
  """Diversity promoting sampling method that uses graph density to determine
  most representative points.
  """

  # def __init__(self, X, y, seed, gamma=None, importance_scores=None, n_neighbor=10, graph_mode='product', graph_sampling_mode='absolute',
  #              precomputed_dists=None, precomputed_neighbors=None):
  def __init__(self, X, y, seed, gamma=None, importance_scores=None, args=None):
    self.name = 'graph_density'
    self.X = X
    if self.X is not None:
      self.flat_X = self.flatten_X()
    # Set gamma for gaussian kernel to be equal to 1/n_features
    if gamma is not None:
      self.gamma = gamma
    else:
      self.gamma = 1. / self.X.shape[1]
    self.graph_mode = args.graph_mode
    self.graph_sampling_mode = args.graph_sampling_mode
    # print("Initializing with gamma value %s and median sampling set to %s" % (self.gamma, self.graph_mode))
    if args.precomputed_dists and args.precomputed_neighbors:
      self.precomputed = True
      self.initialize_with_precomputed_graph(args.precomputed_dists, args.precomputed_neighbors, importance_scores, n_neighbor=args.n_neighbor)
    else:
      self.precomputed = False
      self.compute_graph_density(n_neighbor=args.n_neighbor, importance_scores=importance_scores)

  def initialize_with_precomputed_graph(self, precomputed_dists, precomputed_neighbors, importance_scores, n_neighbor):
    epsilon = 0.0000001
    top_k_distances, top_k_indices = np.load(precomputed_dists)[:, 1:n_neighbor+1], np.load(precomputed_neighbors)[:, 1:n_neighbor+1]
    print("Distances, indices: ", top_k_distances.shape, top_k_indices.shape)
    start_time = time.time()
    importance_scores = importance_scores.numpy()
    self.connect = np.exp(-top_k_distances)*importance_scores[top_k_indices]
    self.distances = top_k_distances
    self.neighbors = top_k_indices
    if self.graph_mode == 'sum':
      self.graph_density = np.sum(self.connect, axis=-1) + importance_scores
    elif self.graph_mode == 'product':
      self.graph_density =  np.sum(self.connect, axis=-1) * importance_scores
    else:
      raise ValueError
    self.starting_density = copy.deepcopy(self.graph_density)
    print("Finished creating graph from precomputed distances in ", time.time() - start_time, "seconds")

  def compute_graph_density(self, n_neighbor=10, importance_scores=None):

    # print("Computing distances for sample with shape:", self.flat_X.shape)
    self.distances = pairwise_distances(self.flat_X, self.flat_X)
    # print("Finished computing distances in ", time.time()-start_time, "seconds")
    if importance_scores is not None and self.graph_mode in ['sum', 'product']:
    # if False:
      epsilon = 0.0000001
      # kneighbors graph is constructed using k=10
      n_samples = self.flat_X.shape[0]
      connect = kneighbors_graph(self.flat_X, n_neighbor,p=2)
      connect = connect.todense()
      # median_distance = np.median(np.reshape(connect, (n_samples*n_samples, ))[0, n_samples:], axis=-1).item()
      # mask = np.array(connect < median_distance, dtype=int)
      # connect = np.multiply(connect, mask)
      # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
      # another point, make it vice versa
      neighbors = connect.nonzero()
      inds = zip(neighbors[0], neighbors[1])
      print("%s connected nodes" % len(neighbors[0]))
      # Graph edges are weighted by applying gaussian kernel to manhattan dist.
      # By default, gamma for rbf kernel is equal to 1/n_features but may
      # get better results if gamma is tuned.
      for entry in inds:
        i = entry[0]
        j = entry[1]
        # distance = pairwise_distances(self.flat_X[[i]], self.flat_X[[j]])  # euclidean
        # distance = distance[0, 0]
        distance = self.distances[i, j]
        weight_j = np.exp(-distance) * max(importance_scores[j].item(), epsilon)
        weight_i = np.exp(-distance) * max(importance_scores[i].item(), epsilon)
        connect[i, j] = weight_j
        connect[j, i] = weight_i
      self.connect = connect
      # print(connect)
      # Define graph density for an observation to be sum of weights for all
      # edges to the node representing the datapoint.  Normalize sum weights
      # by total number of neighbors.
      self.graph_density = np.zeros(self.X.shape[0])
      for i in np.arange(self.X.shape[0]):
        if self.graph_mode == 'sum':
          self.graph_density[i] = connect[i, :].sum() + importance_scores[i].item()
        elif self.graph_mode == 'product':
          self.graph_density[i] = connect[i, :].sum() * importance_scores[i].item()
        else:
          raise ValueError
      self.starting_density = copy.deepcopy(self.graph_density)

    elif importance_scores is not None and self.graph_mode == 'median':
      epsilon = 0.0000001
      # kneighbors graph is constructed using k=10
      n_samples = self.flat_X.shape[0]
      connect = kneighbors_graph(self.flat_X, n_neighbor,p=2, mode='distance')
      connect = connect.todense()
      print(connect, connect.shape)
      median_distance = np.median(np.reshape(connect, (n_samples*n_samples, ))[0, n_samples:], axis=-1).item()
      print(median_distance)
      mask = np.array(connect < median_distance, dtype=int)
      print(mask, np.sum(mask))
      connect = np.multiply(connect, mask)
      # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
      # another point, make it vice versa
      weights = np.tile(importance_scores, (n_samples, 1))
      weights = weights + np.tile(np.transpose(np.expand_dims(importance_scores, axis=0)), (1,n_samples))
      weights = np.maximum(weights, np.ones((n_samples, n_samples))*epsilon)
      connect = np.divide(connect, weights) * -1
      connect = np.exp(connect)
      self.connect = np.multiply(connect, mask)
      # Define graph density for an observation to be sum of weights for all
      # edges to the node representing the datapoint.  Normalize sum weights
      # by total number of neighbors.
      self.graph_density = np.squeeze(np.asarray(np.multiply(np.squeeze(np.sum(connect, axis=-1)), importance_scores)))
      self.starting_density = copy.deepcopy(self.graph_density)

    else:
      # kneighbors graph is constructed using k=10
      connect = kneighbors_graph(self.flat_X, n_neighbor,p=2)
      # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
      # another point, make it vice versa
      neighbors = connect.nonzero()
      inds = zip(neighbors[0],neighbors[1])
      connect = connect.todense()
      # Graph edges are weighted by applying gaussian kernel to manhattan dist.
      # By default, gamma for rbf kernel is equal to 1/n_features but may
      # get better results if gamma is tuned.
      for entry in inds:
        i = entry[0]
        j = entry[1]
        # distance = pairwise_distances(self.flat_X[[i]],self.flat_X[[j]]) # euclidean
        # distance = distance[0,0]
        distance = self.distances[i,j]
        weight = np.exp(-distance * self.gamma)
        connect[i,j] = weight
        connect[j,i] = weight
      self.connect = connect
      # Define graph density for an observation to be sum of weights for all
      # edges to the node representing the datapoint.  Normalize sum weights
      # by total number of neighbors.
      self.graph_density = np.zeros(self.X.shape[0])
      for i in np.arange(self.X.shape[0]):
        self.graph_density[i] = connect[i,:].sum() / (connect[i,:]>0).sum()
      self.starting_density = copy.deepcopy(self.graph_density)

  def select_batch_from_precomputed_(self, N, **kwargs):
    # If a neighbor has already been sampled, reduce the graph density
    # for its direct neighbors to promote diversity.
    batch = set()
    # self.graph_density[already_selected] = min(self.graph_density) - 1
    select = np.zeros(self.graph_density.shape[0])
    min_score = np.min(self.graph_density)
    while len(batch) < N:

      selected = np.argmax(self.graph_density)
      if select[selected] == 1:
        self.graph_density[selected] = min_score - 1
        min_score = min_score - 1
        continue
      else:
        select[selected] = 1

      neighbors = self.neighbors[selected]
      if self.graph_sampling_mode == 'absolute':
        self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
      elif self.graph_sampling_mode =='weighted':
        self.graph_density[neighbors] = self.graph_density[neighbors] - np.exp(-self.distances[selected]*self.gamma)*self.graph_density[selected]
      else:
        raise ValueError
      batch.add(selected)

      # print('(', selected, ',', round(self.graph_density[selected], 2), ')', end=' | ')
      min_score = min(min_score, np.min(self.graph_density[neighbors]))
      # self.graph_density[list(batch)] = min_score - 1
      if len(batch) % 5000 == 0:
        print("%s/%s" % (len(batch), N))
    return list(batch)

  def select_batch_(self, N, **kwargs):

    if self.precomputed:
      batch = self.select_batch_from_precomputed_(N, **kwargs)
    else:
      # If a neighbor has already been sampled, reduce the graph density
      # for its direct neighbors to promote diversity.
      batch = set()
      # self.graph_density[already_selected] = min(self.graph_density) - 1
      while len(batch) < N:
        selected = np.argmax(self.graph_density)
        if type(self.connect) == dict:
          pass
        else:
          neighbors = (self.connect[selected,:] > 0).nonzero()[1]
        if self.graph_sampling_mode == 'absolute':
          self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
        elif self.graph_sampling_mode =='weighted':
          self.graph_density[neighbors] = self.graph_density[neighbors] - np.exp(-self.distances[selected, neighbors]*self.gamma)*self.graph_density[selected]
        else:
          raise ValueError
        batch.add(selected)
        # print('(', selected, ',', round(self.graph_density[selected], 2), ')', end=' | ')
        self.graph_density[list(batch)] = min(self.graph_density) - 1
    return list(batch)


  def to_dict(self):
    output = {}
    output['connectivity'] = self.connect
    output['graph_density'] = self.starting_density
    return output
