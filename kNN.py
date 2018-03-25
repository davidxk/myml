import numpy as np

def euclidean(data, x):
    return np.linalg.norm(data - x, axis = 1)

class kNN:
    def __init__(self, k, distance_function=euclidean):
        """ distance_function has to be able to work on matrix and vector """
        self.k = k
        self.dist = distance_function

    def train(self, X, y):
        self.data = X
        self.labels = y
    
    def predict(self, X):
        y = []
        for x in X:
            dists = self.dist(self.data, x)
            # The first k indices is for the smallest k distances
            partitioned = np.argpartition(dists, self.k)
            voters = partitioned[:self.k]
            # Take a vote
            values, cnts = np.unique(self.labels[voters], return_counts=True)
            y.append( values[np.argmax(cnts)] )
        return np.array(y)
