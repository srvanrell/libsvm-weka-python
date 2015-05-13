#!/usr/bin/python

import os

mfilename = 'temp.m'
mfile = open(mfilename, 'w')
mfile.write("""
conteo = 1:10
magica = magic(5)
""")
mfile.close()

os.system("matlab -nojvm < test.m > salida.txt 2> error.txt")

os.remove(mfilename)