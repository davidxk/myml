# Tasks
* Gather data
* Design project scope
* Design classes and interfaces

## Gather Data
Classification: 

* Binary
* Multi-class

Regression:

* Linear Regression
* Nonlinear basis

It appears most raw real-world data is not fit for direct use. 

## Design Scope
* Regression
    * Linear Regression: pending regression data
* Classification
    * kNN
    * Binary Logistic Regression
    * Binary Perceptron
    * Multi-class Logistic Regression
	* SVM
* Clustering
    * k-means

## Classes & Interfaces
class Supervised learner

* train(self, X, y)
* test(self, X)

class Runner

* loadData(self)
* run(self, learner)
* evaluate(self, true, pred)

# Develop Stages
## Stage 1 –– Done
+ Implement Linear Regression
+ Implement Logistic Regression
+ Implement kNN
+ Implement loadData for regression, binary classification, multi-class classification
+ Write main program to run the three algorithms

## Stage 2 –– Done
+ Implement Perceptron
+ Implement Multinomial Logistic Regression
+ Find linearly separable data for perceptron

## Stage 4 –– Done
+ Implement Multinomial Perceptron
+ Write separable classification runner

## Stage 4 –– Done
+ Mini-batch gradient descent
