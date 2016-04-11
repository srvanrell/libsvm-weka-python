#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script to test that needed package are correctly installed
Tested with:
- python-weka-wrapper 3.6.0
- LibSVM 1.0.6
"""

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation, GridSearch

jvm.logger.setLevel(jvm.logging.WARNING)
jvm.start(packages=True, max_heap_size="512m")

# Each instance has nominal class and numeric attributes
loader = Loader(classname="weka.core.converters.ArffLoader")
trainData = loader.load_file('iris.arff')
trainData.class_is_last()

# LibSVM with RBF kernel, cost C and gamma should be optimized
classifier = Classifier(classname="weka.classifiers.functions.LibSVM",
                        options=["-S", "0", "-K", "2", "-Z", "-G", "0.0", "-C", "1.0",
                                 "-D", "3", "-R", "0.0", "-N", "0.5", "-M", "40.0",
                                 "-E", "0.001", "-P", "0.1", "-model", "~/", "-seed", "1"])

# Logarithmic grid search on C and gamma, with cross validation (mandatory) on the training set
grid = GridSearch(options=["-sample-size", "100.0", "-traversal", "ROW-WISE", "-num-slots", "1", "-S", "1",
                           "-output-debug-info"
                           ])
grid.evaluation = "ACC"
grid.x = {"property": "cost", "min": -3.0, "max": 3.0, "step": 1.0, "base": 10.0, "expression": "pow(BASE,I)"}
grid.y = {"property": "gamma", "min": -3.0, "max": 3.0, "step": 1.0, "base": 10.0, "expression": "pow(BASE,I)"}

# LibSVM is added to grid configuration
grid.classifier = classifier
# Search for the best parameters and build a classifier with them
grid.build_classifier(trainData)
best = grid.best
best.build_classifier(trainData)

print grid
print best

# Evaluation on train dataset just to simplify this example
evaluation = Evaluation(trainData)
evaluation.test_model(best, trainData)
print best.to_commandline()
print evaluation.matrix()

jvm.stop()
