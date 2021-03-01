.. index:: mesh; surface elevations

Fitting surface elevation data
==============================

Layermesh meshes may have incomplete upper layers (i.e. different
columns may have different numbers of layers) to represent
e.g. surface topography. The surface of the mesh can be specified by
fitting arbitrary scattered (*x*, *y*, *z*) data, using the ``mesh``
``fit_surface()`` method.

This method uses least-squares finite element fitting with piecewise
constant elements to determine an appropriate surface elevation for
each column. The number of layers in the column is then determined by
taking this fitted elevation and choosing the nearest layer boundary
as the top surface of the column.

On its own, however, this algorithm will fail if the dataset is sparse
and there are columns which do not contain any data points. To
overcome this (and also to help overcome problems with noisy data) an
additional smoothing term is introduced to the least-squares fitting
process. This term is simply the sum of squares of the differences in
elevation across the faces between columns. This term is weighted by a
``smoothing`` parameter (with default value 0.01) which may be passed
into the ``fit_surface()`` method.

For example:

.. code-block:: python

  import layermesh.mesh as lm
  import numpy as np

  m = lm.mesh(rectangular = ([1000]*10, [800]*12, [100]*8))
  surf = np.loadtxt('surface.txt')
  m.fit_surface(surf, smoothing = 0.02)

creates a simple rectangular mesh, loads surface elevation data from a
text file containing (*x*, *y*, *z*) data on each line, and fits the
mesh surface to the data using a smoothing parameter of 0.02.

Generally only a small value of the smoothing parameter is needed to
overcome problems with sparse data. Its value can be increased if the
dataset is noisy and there are large gradients in the fitted surface.

It is also possible to fit data over only some of the mesh columns,
rather than all of them (the default). To do this, the ``columns``
parameter is used, which takes a tuple or list of columns to be
fitted:

.. code-block:: python

  cols = m.find([(0,0), (5000, 5000)])
  m.fit_surface(surf, columns = cols, smoothing = 0.02)

Here surface fitting is carried out for all columns with centroids
within a rectangle with bottom left coordinates at the origin and top
right coordinates (5000, 5000). (For more information on how to find
particular mesh columns or other mesh components using the ``find()``
method, see :ref:`searching`.)
