import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=10000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        y_ = np.where(Y <= 0, -1, 1)
        # init weights
        self.w = np.zeros(n_features)
        self.b = 0 
        for _ in range(self.n_iters):
            for idx, X_i in enumerate(X):
                if (y_[idx] * (np.dot(X_i, self.w) - self.b) >= 1):
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
