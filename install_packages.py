#!/usr/bin/python
# -*- coding: utf-8 -*-

import weka.core.jvm as jvm
import weka.core.packages as packages

jvm.start()

# checking for installed packages
#installed_packages = packages.installed_packages()
#for item in installed_packages:
#    print item.name, item.url, "is installed\n"

# # Search for GridSearch and LibSVM, just to check package's names
# all_packages = packages.all_packages()
# for item in all_packages:
#     if (item.name == "gridSearch") or (item.name == "LibSVM"):
#         print(item.name + " " + item.url)

# To install gridSearch and LibSVM
# packages.install_package("gridSearch", "1.0.8")
#packages.install_package("LibSVM")
packages.install_package("/home/sebastian/Descargas/LibSVM1.0.8.zip")

# To install MultiSearch
#packages.install_package("https://github.com/fracpete/multisearch-weka-package/releases/download/" +
#                         "v2016.6.7/multisearch-2016.6.7.zip")
#packages.install_package("/home/sebastian/Descargas/multisearch-2016.6.7.zip")
#packages.uninstall_package("multisearch")

jvm.stop()
