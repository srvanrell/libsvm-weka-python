#!/usr/bin/python

from pymatbridge import Matlab

mlab = Matlab()
mlab.start()
print "Matlab started?", mlab.started
print "Matlab is connected?", mlab.is_connected()

mlab.run_code("conteo = 1:10")
mlab.run_code("magica = magic(5)")

mlab.stop()

