from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances

def get_aucpr(coreset, target):
    # step 1, get L2 distance between embeddings
    n_dim = target.shape[1]
    # if target.shape[0] == 0:
    #     print(target)
        # target = np.expand_dims(target, axis=0)

    if coreset.shape[0] == n_dim:
        coreset = np.expand_dims(coreset, axis=0)
    print("Computing AUCpr between %s and %s samples" % (coreset.shape[0], target.shape[0]))

    # print(target.shape, coreset.shape)
    # target = np.expand_dims(target, axis=0)
    # target = np.broadcast_to(target, (n_coreset, n_target, n_dim))
    # coreset = np.broadcast_to(coreset, (n_target, n_coreset, n_dim))
    # target = np.transpose(target, (1, 0, 2))
    # dist = np.linalg.norm(target-coreset, axis=-1)

    min_dists = []
    for i in tqdm(range(target.shape[0])):
        dist = pairwise_distances(np.expand_dims(target[i], axis=0), coreset)
        min_dists.append(np.amin(dist))
    aucpr = np.sum(min_dists)/target.shape[0]
    return aucpr