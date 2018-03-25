import numpy as np

class MulticlassPerceptron:
    def train(self, X, y):
        """ Assume y in [0, C - 1] """
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))
        C = len(np.unique(y))
        self.w = np.zeros((C, D+1))

        i, cnt = 0, 0
        while cnt < N:
            n = i % N
            xn, yn = X[n], y[n]
            fx = np.argmax( np.dot(xn, self.w.T) ) 
            if fx != yn:
                self.w[fx] -= xn
                self.w[yn] += xn
                cnt = 0
            else:
                cnt += 1
            i += 1

    def predict(self, X):
        N, D = X.shape
        X = np.column_stack((np.ones(N), X))

        y = np.dot(X, self.w.T).argmax(axis = 1)
        return y
