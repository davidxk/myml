import numpy as np

class LogisticRegression:
    def __init__(self):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def bound(self, vector, limit=100):
        """ bound abs of vector elems to prevent overflow """
        vector[vector > limit] = limit
        vector[vector < -limit] = -limit

    def BGD(self, X, y, eta=0.1, max_iteration=1000, epsilon=0.001):
        """
        Logistic Regression - Batch update
        a = Xw
        f(X) = sigmoid(a).round()
        l' = (f(X) - y) X
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)

        for i in range(max_iteration):
            a = np.dot(X, self.w)
            self.bound(a)
            sigm = self.sigmoid(a)
            grad = np.dot(sigm - y, X) / N
            if np.linalg.norm(grad) < epsilon:
                break
            self.w -= eta * grad

    def MGD(self, X, y, B=40, eta=0.1, max_iteration=2000, epsilon=0.001):
        """
        Logistic Regression - Mini-batch
        B for batch size
        a = Xb w
        f(Xb) = sigmoid(a).round()
        l' = (f(Xb) - y) Xb
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)

        for i in range(max_iteration):
            b = np.random.choice(range(N), B)
            Xb, yb = X[b], y[b]
            a = np.dot(Xb, self.w)
            self.bound(a)
            sigm = self.sigmoid(a)
            grad = np.dot(sigm - yb, Xb) / B
            self.w -= eta * grad

    def SGD(self, X, y, eta=0.5, max_iteration=5000, epsilon=0.001):
        """
        Logistic Regression - Stochastic Gradient Descent
        a = Xw
        f(x) = sigmoid(a).round()
        l(xn)' = (f(xn) - yn) xn
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)

        for i in range(max_iteration):
            n = np.random.choice(range(N))
            xn, yn = X[n], y[n]
            a = np.dot(xn, self.w)
            a = a if np.abs(a) < 100 else np.sign(a) * 100
            sigm = self.sigmoid(a)
            grad = np.dot(sigm - yn, xn)
            self.w -= eta * grad

    def train(self, X, y, impl=None):
        assert len(np.unique(y)) == 2
        if impl == "Batch-update":
            self.BGD(X, y)
        elif impl == "Mini-batch":
            self.MGD(X, y)
        else:
            self.SGD(X, y)

    def predict(self, X):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))

        a = np.dot(X, self.w)
        self.bound(a)
        y = self.sigmoid(a)
        y = np.where(y < 0.5, 0, 1)
        return y
