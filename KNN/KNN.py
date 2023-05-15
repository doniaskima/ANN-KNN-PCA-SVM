import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance
class KNN:
    def __init__(self, k=3):
        self.k=k 
    def fit(self, X, Y): 
        self.X_train=X
        self.Y_train=Y 
    def predict(self, X):
        predictions =[self._predict(x) for x in X]
        return predictions
    def _predict(self, x):
        #compute the distance
        distances=[euclidean_distance(x, x_train) for x_train in self.X_train]
        #get the closest k
        k_indices = np.argsort(distances)[:self.k]
        # np.argsort function to obtain the indices of the self.k nearest neighbors.
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        # majority vote
        most_common= Counter(k_nearest_labels).most_common()
        return most_common[0][0]