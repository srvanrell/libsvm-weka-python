#!/usr/bin/python
# -*- coding: utf-8 -*-

import weka.core.jvm as jvm
import weka.core.packages as packages

jvm.start()

installed_packages = packages.installed_packages()
for item in installed_packages:
    print item.name, item.url, "installed"

# Search for GridSearch and LibSVM
all_packages = packages.all_packages()
for item in all_packages:
    if item.get_name() == "gridSearch" or item.get_name() == "LibSVM":
        print(item.name + " " + item.url)

# packages.install_package("Name")
# packages.uninstall_package("Name")

# To install MultiSearch
packages.install_package("http://github.com/fracpete/multisearch-weka-package/releases/download/v2014.12.10/multisearch-2014.12.10.zip")

jvm.stop()