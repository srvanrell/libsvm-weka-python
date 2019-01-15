#!/usr/bin/python
# -*- coding: utf-8 -*-

import weka.core.jvm as jvm
from weka.core.converters import Loader

jvm.logger.setLevel(jvm.logging.INFO)
jvm.start()

loader = Loader(classname="weka.core.converters.ArffLoader")
trainData = loader.load_file('segment-challenge.arff')
testData = loader.load_file('this-file-not-exist')  # silent fail

print(trainData.num_instances)  # True answer (1500)
print(testData.num_instances)   # False answer (1500)

jvm.stop()
