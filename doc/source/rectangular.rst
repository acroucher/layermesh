Creating rectangular meshes
===========================

A simple rectangular ``mesh`` object can be created by using the
``rectangular`` parameter. This is a list or tuple of the mesh
spacings in each coordinate direction. Each mesh spacing specification
is itself a list, tuple or array of spacings.

For example:

.. code-block:: python

  from layermesh import mesh as lm
  m = lm.mesh(rectangular = ([1000]*10, [800]*12, [100]*8))

creates a simple regular rectangular 10×12×8 cell mesh, with constant
mesh spacings in the *x*-, *y*- and *z*-directions of 1000, 800 and
100 respectively.

Irregular rectangular meshes can be created by passing non-uniform
mesh spacings in in the ``rectangular`` parameter. For example:

.. code-block:: python

  from layermesh import mesh as lm
  import numpy as np

  dx = np.arange(1000, 7000, 1000)
  dy = dx
  dz = np.arange(10, 60, 10)
  m = lm.mesh(rectangular = [dx, dy, dz])

creates an irregular rectangular mesh with equal spacings in the *x*-
and *y*-directions ranging from 1000 to 6000, and with layer
thicknesses ranging from 10 at the top to 50 at the bottom.
