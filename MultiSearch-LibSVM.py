#!/usr/bin/env python
"""
Script to test that needed packages are correctly installed
Tested with:
- python-weka-wrapper 0.3.8
- LibSVM 1.0.8
- MultiSearch 2016.6.7
"""

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, MultiSearch, Evaluation
from weka.core.classes import MathParameter

jvm.logger.setLevel(jvm.logging.WARNING)
jvm.start(packages=True, max_heap_size="512m")

# Each instance has nominal class and numeric attributes
loader = Loader(classname="weka.core.converters.ArffLoader")
trainData = loader.load_file('segment-challenge.arff')
trainData.class_is_last()
testData = loader.load_file('segment-test.arff')
testData.class_is_last()

# LibSVM with RBF kernel, cost C and gamma should be optimized
classifier = Classifier(classname="weka.classifiers.functions.LibSVM",
                        options=["-S", "0", "-K", "2", "-Z", "-G", "0.0", "-C", "1.0",
                                 "-D", "3", "-R", "0.0", "-N", "0.5", "-M", "40.0",
                                 "-E", "0.001", "-P", "0.1", "-model", "/home", "-seed", "1"])

# Logarithmic grid search on C and gamma, without cross validation on the training set
gamma = MathParameter()
gamma.prop = "gamma"
gamma.minimum, gamma.maximum, gamma.step, gamma.base = -3.0, 5.0, 1.0, 10.0
gamma.expression = "pow(BASE,I)"

cost = MathParameter()
cost.prop = "cost"
cost.minimum, cost.maximum, cost.step, cost.base = -3.0, 5.0, 1.0, 10.0
cost.expression = "pow(BASE,I)"

grid = MultiSearch(options=["-sample-size", "100.0", "-num-slots", "8", "-S", "1",
                            "-initial-folds", "2", "-subsequent-folds", "3",
                            "-output-debug-info"
                            ])
grid.classifier = classifier
grid.evaluation = "ACC"
grid.parameters = [gamma, cost]

# LibSVM is added to grid configuration
grid.classifier = classifier
# Search for the best parameters and build a classifier with them
grid.build_classifier(trainData)
best = grid.best
best.build_classifier(trainData)

print(best.options)
print("C", best.options[best.options.index("-C")+1])
print("gamma", best.options[best.options.index("-G")+1])

print("\n\n=========== Train results ================\n\n")
print(grid)
evaluation = Evaluation(trainData)
evaluation.test_model(best, trainData)
print(best.to_commandline())
print(evaluation.matrix())
print("Train recognition: %0.2f%%" % evaluation.percent_correct)

print("\n\n=========== Test results ================\n\n")
evaluation = Evaluation(testData)
evaluation.test_model(best, testData)
print(best.to_commandline())
print(evaluation.matrix())
print("Test recognition: %0.2f%%" % evaluation.percent_correct)

jvm.stop()
