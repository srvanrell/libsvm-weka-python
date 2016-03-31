#!/usr/bin/python
# -*- coding: utf-8 -*-

import weka.core.jvm as jvm
from weka.core.converters import Loader

jvm.logger.setLevel(jvm.logging.INFO)
jvm.start()

# Each instance has nominal class and numeric attributes
loader = Loader(classname="weka.core.converters.ArffLoader")
trainData = loader.load_file('segment-challenge.arff')
testData = loader.load_file('this-file-not-exist')  # silent fail

print(trainData.num_instances)  # True answer
print(testData.num_instances)   # False answer

jvm.stop()
