#!/usr/bin/python
# -*- coding: utf-8 -*-

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation, SingleClassifierEnhancer
import javabridge

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
                                 "-E", "0.001", "-P", "0.1", "-model", "~/", "-seed", "1"])

# Logaritmic grid search on C and gamma, without cross validation on the training set
grid = SingleClassifierEnhancer(
    classname="weka.classifiers.meta.MultiSearch",
    options=[
        "-E", "ACC",
        "-search", 'weka.core.setupgenerator.MathParameter -property classifier.cost -min -3.0 -max 5.0 -step 1.0 -base 10.0 -expression "pow(BASE,I)"',
        "-search", 'weka.core.setupgenerator.MathParameter -property classifier.gamma -min -3.0 -max 5.0 -step 1.0 -base 10.0 -expression "pow(BASE,I)"',
        "-output-debug-info",
        "-initial-folds", "2", "-subsequent-folds", "3",
        "-W", "weka.classifiers.trees.J48",  # dummy classifier added to avoid Nominal Class Capability Exception
        "-sample-size", "100.0", "-num-slots", "8", "-S", "1"])

# LibSVM is added to grid configuration
grid.classifier = classifier
# Search for the best parameters and build a classifier with them
grid.build_classifier(trainData)
best = Classifier(jobject=javabridge.call(grid.jobject, "getBestClassifier", "()Lweka/classifiers/Classifier;"))
best.build_classifier(trainData)

print best.options
print "C", best.options[best.options.index("-C")+1]
print "gamma", best.options[best.options.index("-G")+1]
# TODO install new release 3.0.1 and use new method to define a finer grid


print "\n\n=========== Train results ================\n\n"
print grid
evaluation = Evaluation(trainData)
evaluation.test_model(best, trainData)
print best.to_commandline()
print evaluation.matrix()
print "Train recognition: %0.2f%%" % evaluation.percent_correct

print "\n\n=========== Test results ================\n\n"
evaluation = Evaluation(testData)
evaluation.test_model(best, testData)
print best.to_commandline()
print evaluation.matrix()
print "Test recognition: %0.2f%%" % evaluation.percent_correct

jvm.stop()
