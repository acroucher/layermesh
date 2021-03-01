.. index:: mesh; plotting

Creating 2-D plots
==================

Layermesh can be used to create 2-D plots of the mesh, with cells
optionally labelled and/or shaded with values (e.g. simulation
results).

Layermesh ``mesh`` objects have two methods for creating plots:

* the ``layer_plot()`` method creates a plot over a specified mesh layer
* the ``slice_plot()`` method creates a plot over a specified vertical
  slice through the mesh

In either case, the `Matplotlib <https://matplotlib.org/>`_ library is
used to create the plot, which can be either viewed directly on the
display or saved to an image file.

.. index:: mesh; layer plots

Layer plots
-----------

The ``mesh`` ``layer_plot()`` method takes as its first parameter the
layer to be plotted - either a ``layer`` object, or an integer mesh
layer index. Alternatively, an elevation can be specified via the
``elevation`` parameter, which will then be used to determine the
appropriate layer. If neither the layer or an elevation is specified,
then the bottom layer is plotted.

Examples:

.. code-block:: python

  m.layer_plot() # plot bottom layer

  lay = m.layer[2]
  m.layer_plot(lay) # plot layer 2

  m.layer_plot(elevation = -1350) # plot layer containing elevation -1350

.. index:: mesh; slice plots

Slice plots
-----------

The ``mesh`` ``slice_plot()`` method takes as its first parameter the
line defining the slice to be plotted. This can be either:

* a string "x" or "y" to plot through the mesh centre along the *x*- or *y*-axes
* a number representing an angle (in degrees clockwise from the
  *y*-axis) to plot through the mesh centre on that angle
* a tuple, list or array of two 2-D points representing the end-points
  of the line

Examples:

.. code-block:: python

  m.slice_plot() # plot through centre along x-axis

  m.slice_plot('y') # plot through centre along y-axis

  m.slice_plot(45) # plot through centre at 45 degrees from y-axis

  line = [(0,0), (3000, 4000)]
  m.slice_plot(line) # plot along specified line

Plotting values over the mesh
-----------------------------

Both ``layer_plot()`` and ``slice_plot()`` take an optional ``value``
parameter, which is a tuple, list or rank-1 array of values to plot
over the mesh. The length of the ``value`` parameter should be equal
to the number of mesh cells.  For example:

.. code-block:: python

  T = np.loadtxt('temperatures.txt')
  m.layer_plot(elevation = -50, value = T)

loads an array of values from a text file and plots them over the
layer at elevation -50.

When a value is plotted, a colourbar scale is drawn next to the
plot. The optional ``value_label`` and ``value_unit`` parameters can
be used to produce the name of the quantity being plotted on the
colourbar, together with its units, e.g.:

.. code-block:: python

  m.layer_plot(elevation = -50, value = T,
    value_label = 'Temperature', value_unit = 'deg C')

Plotting labels
---------------

The ``layer_plot()`` and ``slice_plot()`` methods also have an
optional ``label`` parameter, if labels are to be drawn at the centre
of each cell in the plot.

The ``label`` parameter is a string and can be either:

* "cell": label cells with cell indices
* "value": label cells with numerical values, taken from the ``value``
  parameter
* "column" (``layer_plot()`` only): label cells with column indices

Examples:

.. code-block:: python

  m.slice_plot('x', label = 'cell') # plot along x-axis, labelling cell indices

  m.layer_plot(10, label = 'column') # plot layer 10, labelling column indices

  m.slice_plot('y', value = T, label = 'value') # plot and label T along y-axis

Plot output
-----------

By default, the ``layer_plot()`` and ``slice_plot()`` methods plot
directly to the display, so a plot will appear immediately after the
method is called.

It is also possible to plot to a Matplotlib ``axes`` object instead,
via the ``axes`` parameter of the ``layer_plot()`` and
``slice_plot()`` methods. This can be useful for e.g.:

* putting multiple plots on one page
* superimposing other things on the plot
* saving the output to an image file

For example:

.. code-block:: python

  import layermesh.mesh as lm
  import numpy as np
  import matplotlib.pyplot as plt

  m = lm.mesh('mymesh.h5')
  P = np.loadtxt('pressures.txt')
  T = np.loadtxt('temperatures.txt')

  fig = plt.figure()

  ax = fig.add_subplot(2, 1, 1)
  m.slice_plot('x', axes = ax, value = P,
    value_label = 'Pressure', value_unit = 'bar')

  ax = fig.add_subplot(2, 1, 2)
  m.slice_plot('x', axes = ax, value = T,
    value_label = 'Temperature', value_unit = 'deg C')

  plt.suptitle('Pressure and temperature plots along x-axis')
  plt.savefig('plots.png')

Here a mesh is loaded from an HDF5 file, along with the datasets ``P``
and ``T`` which are loaded from text files. A Matplotlib figure is
created, and within it, axes for two subplots. These are used to call
``slice_plot()`` twice, to plot ``P`` and ``T`` along an *x*-axis
slice.

Finally, the plot is given a title and the output saved to an image file.

If the ``axes`` parameter is passed to ``layer_plot()`` or
``slice_plot()``, nothing will appear on the display when the method
is called. In the above example the plot could be shown by adding:

.. code-block:: python

  plt.show()
