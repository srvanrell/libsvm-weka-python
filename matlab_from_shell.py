#!/usr/bin/python

import os

mfilename = 'temp.m'
mfile = open(mfilename, 'w')
mfile.write("""
conteo = 1:10
magica = magic(5)
""")
mfile.close()

os.system("matlab -nojvm < {0:s} > salida.txt 2> error.txt".format(mfilename))

os.remove(mfilename)