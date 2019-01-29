# Cython compile instructions

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = "PredFunction: a part of farbopt",
    ext_modules = cythonize('farbopt.pyx')
)
