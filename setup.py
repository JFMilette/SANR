"""
Builds the compiled transfer-matrix kernel (model/_transfer.pyx) in place:

    python setup.py build_ext --inplace

Needs Cython and a C compiler.  Without the build model.stack uses its
numpy version of the same product (slower, same result); build again after
editing the .pyx.
"""

import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

# -fcx-limited-range: plain complex * and /, no C99 inf/NaN recovery call
# per operation (the kernel never multiplies infinities on purpose)
ARGS = [] if sys.platform == 'win32' else ['-O3', '-fcx-limited-range']

setup(
    name='sanr-kernels',
    ext_modules=cythonize(
        [Extension('model._transfer', ['model/_transfer.pyx'],
                   extra_compile_args=ARGS)],
        language_level=3),
)
