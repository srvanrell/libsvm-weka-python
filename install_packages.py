#!/usr/bin/python
# -*- coding: utf-8 -*-

import weka.core.jvm as jvm
import weka.core.packages as packages

jvm.start()

# To install gridSearch and LibSVM
packages.install_package("gridSearch", "1.0.8")
packages.install_package("LibSVM")
# To install MultiSearch
packages.install_package("https://github.com/fracpete/multisearch-weka-package/releases/download/v2014.12.10/multisearch-2014.12.10.zip")

jvm.stop()