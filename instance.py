import numpy as np
from scipy.spatial.distance import pdist, squareform


class Instance:

    def __init__(self, X, k, color):
        self.X = X
        n = X.shape[0]
        self.n = n
        self.k = k
        self.color = color 
        self.distance = squareform(pdist(X, 'sqeuclidean'))