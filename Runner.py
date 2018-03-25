from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import numpy as np

class Runner:
    def loadData(self):
        pass

    def run(self, learner, arg=None):
        X_train, X_test, y_train, y_test = self.loadData()
        learner.train(X_train, y_train, arg) if arg \
                else learner.train(X_train, y_train)
        pred = learner.predict(X_test)
        name, value = self.evaluate(y_test, pred)
        print(name, value)

    def evaluate(self, true, pred):
        pass

class RegressionRunner(Runner):
    def loadData(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                random_state=3)
        return X_train, X_test, y_train, y_test

    def evaluate(self, true, pred):
        mean = true.mean()
        r2 = np.sum((pred - mean) ** 2) / np.sum((true - mean) ** 2)
        return ("R2", r2)

    def mse(self, true, pred):
        assert(len(true) == len(pred))
        mse = np.mean( (pred - true) ** 2 )
        return ("MSE", mse)

class ClassificationRunner(Runner):
    def loadData(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                random_state=3)
        return X_train, X_test, y_train, y_test

    def evaluate(self, true, pred):
        assert(len(true) == len(pred))
        diff = (pred - true) == 0
        accu = np.count_nonzero(diff) / len(true)
        return ("Accuracy", accu)

class BinaryClassificationRunner(ClassificationRunner):
    def loadData(self, labels=(0, 1)):
        X, y = load_breast_cancer(return_X_y=True)

        assert(len(labels) == 2)
        if labels != (0, 1):
            y = np.where(y == 0, labels[0], labels[1])

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                random_state=3)
        return X_train, X_test, y_train, y_test

class PerceptronRunner(ClassificationRunner):
    def __init__(self, n=2):
        self.n = n

    def loadData(self):
        X, y = make_classification(n_samples=self.n * 50, n_features=self.n,
                n_informative=self.n, n_redundant=0, n_classes=self.n, flip_y=0.00,
                class_sep=self.n * 2, random_state=500)
        if self.n == 2:
            y = 2 * y - 1

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                random_state=3)
        return X_train, X_test, y_train, y_test

class ClusteringRunner(Runner):
    def loadData(self):
        data = load_digits()
        X, y = data.data, data.target
        k = len(data.target_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                random_state=42)
        return X_train, k, X_test, y_test

    def run(self):
        X_train, X_test, y_test = self.loadData()
        learner.train(X_train, k)
        pred = learner.predict(X_test)
        name, value = self.evaluate(y_test, pred)
        print(name, value)

    def evaluate(self, true, pred):

        return ("Adjusted Rand index", ari)
