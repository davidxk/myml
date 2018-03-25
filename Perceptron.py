import numpy as np

class Perceptron:
    def train(self, X, y):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        self.w = np.zeros(D + 1)

        i, cnt = 0, 0
        while cnt < N:
            n = i % N
            xn, yn = X[n], y[n]
            fx = np.sign( np.dot(xn, self.w) ) 
            if fx != yn:
                self.w += yn * xn
                cnt = 0
            else:
                cnt += 1
            i += 1

    def predict(self, X):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        y = np.sign(np.dot(X, self.w))
        return y
