import numpy as np

class LinearRegression:
    def closedForm(self, X, y):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        XT = X.T
        self.w = np.dot(np.linalg.pinv(XT.dot(X)), XT.dot(y))

    def SGD(self, X, y, eta=0.01, max_iteration=1000, epsilon=0.001):
        """
        Stochastic Gradient Descent
        eta: learning rate for stochastic gradient descent
        max_iteration: maximum number of iterations allowed
        epsilon: minimum change observed to keep the iterations going
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)
        for i in range(max_iteration):
            n = np.random.choice(range(N))
            xn, yn = X[n], y[n]
            gn = (np.dot(self.w, xn) - yn) * xn
            #if np.linalg.norm(gn) < epsilon:
                #break
            self.w -= eta * gn

    def MGD(self, X, y, B=40, eta=0.001, max_iteration=2000, epsilon=0.001):
        """
        Mini-batch Gradient Descent
        eta: learning rate for stochastic gradient descent
        max_iteration: maximum number of iterations allowed
        epsilon: minimum change observed to keep the iterations going
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)
        for i in range(max_iteration):
            b = np.random.choice(range(N), B)
            Xb, yb = X[b], y[b]
            gn = (np.dot(Xb, self.w) - yb).dot(Xb) / B
            #if np.linalg.norm(gn) < epsilon:
                #break
            self.w -= eta * gn

    def BGD(self, X, y, eta=0.001, max_iteration=1000, epsilon=0.001):
        """
        Batch-update Gradient Descent
        eta: learning rate for stochastic gradient descent
        max_iteration: maximum number of iterations allowed
        epsilon: minimum change observed to keep the iterations going
        """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)
        for i in range(max_iteration):
            gn = (np.dot(X, self.w) - y).dot(X) / N
            #if np.linalg.norm(gn) < epsilon:
                #break
            self.w -= eta * gn

    def train(self, X, y, impl=None):
        if impl == "Closed-form":
            self.closedForm(X, y)
        elif impl == "Batch-update":
            self.BGD(X, y)
        elif impl == "Mini-batch":
            self.MGD(X, y)
        else:
            self.SGD(X, y)

    def predict(self, X):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        fx = np.dot(X, self.w)
        return fx
