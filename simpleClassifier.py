#!/usr/bin/env python

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation

jvm.logger.setLevel(jvm.logging.WARNING)
jvm.start(packages=True, max_heap_size="512m")

# Each instance has nominal class and numeric attributes
loader = Loader(classname="weka.core.converters.ArffLoader")
trainData = loader.load_file('segment-challenge.arff')
trainData.class_is_last()
testData = loader.load_file('segment-test.arff')
testData.class_is_last()

# Default C4.5 tree
classifier = Classifier(classname="weka.classifiers.trees.J48")

# Search for the best parameters and build a classifier with them
classifier.build_classifier(trainData)

print("\n\n=========== Classifier information ================\n\n")
print(classifier.options)
print(classifier)

print("\n\n=========== Train results ================\n\n")
evaluation = Evaluation(trainData)
evaluation.test_model(classifier, trainData)
print(classifier.to_commandline())
print(evaluation.matrix())
print("Train recognition: %0.2f%%" % evaluation.percent_correct)

print("\n\n=========== Test results ================\n\n")
evaluation = Evaluation(testData)
evaluation.test_model(classifier, testData)
print(classifier.to_commandline())
print(evaluation.matrix())
print("Test recognition: %0.2f%%" % evaluation.percent_correct)

jvm.stop()
