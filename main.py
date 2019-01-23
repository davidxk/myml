from Runner import *
from kNN import kNN
from Perceptron import Perceptron
from MulticlassPerceptron import MulticlassPerceptron
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from MultinomialLogistic import MultinomialLogistic

runner = RegressionRunner()
runner.run(LinearRegression(), "Closed-form")
runner.run(LinearRegression(), "Batch-update")
runner.run(LinearRegression(), "Mini-batch")
runner.run(LinearRegression(), "Stochastic")
#runner = BinaryClassificationRunner()
#runner.run(LogisticRegression(), "Batch-update")
#runner.run(LogisticRegression(), "Mini-batch")
#runner.run(LogisticRegression(), "Stochastic")
#runner = ClassificationRunner()
#runner.run(kNN(5))
#runner.run(MultinomialLogistic(), "Batch-update")
#runner.run(MultinomialLogistic(), "Mini-batch")
#runner.run(MultinomialLogistic(), "Stochastic")
#runner = PerceptronRunner()
#runner.run(Perceptron())
#runner = PerceptronRunner(4)
#runner.run(MulticlassPerceptron())
