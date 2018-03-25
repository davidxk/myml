import numpy as np

class MultinomialLogistic:
    def __init__(self):
        def softmax(a):
            a = a - a.max(axis = 1)[:, None]
            exp = np.exp(a)
            res = exp / exp.sum(axis = 1)[:, None]
            return res
        self.softmax = softmax

    def bound(self, vector, limit=100):
        """ bound abs of vector elems to prevent overflow """
        vector[vector > limit] = limit
        vector[vector < -limit] = -limit

    def BGD(self, X, y, eta=0.1, max_iteration=1000, epsilon=0.001):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        # 1-of-K encoding
        C = len(np.unique(y))
        Y = np.zeros((N, C))
        Y[np.arange(N), y] = 1

        self.w = np.zeros((C, D+1))
        for i in range(max_iteration):
            a = np.dot(X, self.w.T)
            grad = (self.softmax(a) - Y).T.dot(X)
            self.w -= eta * grad / N

    def MGD(self, X, y, B=40, eta=0.1, max_iteration=2000, epsilon=0.001):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        # 1-of-K encoding
        C = len(np.unique(y))
        Y = np.zeros((N, C))
        Y[np.arange(N), y] = 1

        self.w = np.zeros((C, D+1))
        for i in range(max_iteration):
            b = np.random.choice(range(N), B)
            Xb, Yb = X[b], Y[b]
            a = np.dot(Xb, self.w.T)
            grad = (self.softmax(a) - Yb).T.dot(Xb)
            self.w -= eta * grad / B

    def SGD(self, X, y, eta=0.1, max_iteration=2000, epsilon=0.001):
        def softmax(a):
            """ Avoid overflow: e^k/sum == (e^k/e^max) / (sum/e^max) """
            a = a - a.max()
            exp = np.exp(a)
            res = exp / exp.sum()
            return res
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        # 1-of-K encoding
        C = len(np.unique(y))
        Y = np.zeros((N, C))
        Y[np.arange(N), y] = 1

        self.w = np.zeros((C, D+1))
        for i in range(max_iteration):
            n = np.random.choice(range(N))
            xn, yn = X[n], Y[n]
            a = np.dot(xn, self.w.T)
            #self.bound(a)
            grad = np.outer(softmax(a) - yn, xn)
            self.w -= eta * grad

    def train(self, X, y, impl=None):
        if impl == "Batch-update":
            self.BGD(X, y)
        elif impl == "Mini-batch":
            self.MGD(X, y)
        else:
            self.SGD(X, y)

    def predict(self, X):
        """
        [ --- ]
        [ --- ]
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))

        y = np.dot(X, self.w.T).argmax(axis = 1)
        return y
