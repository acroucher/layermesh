.. _searching:

Searching
=========

Searching for particular Layermesh mesh components (e.g. cells,
columns or layers) can be carried out using the ``mesh`` ``find()``
method. This method can be used in several different ways, depending
on what kind of parameters it is given.

Searching for cells
-------------------

Cells can be found based on their 3-D position, or via a
user-specified function to select cells with particular attributes.

Searching by position
.....................

Passing a 3-D point as the parameter to the ``mesh`` ``find()`` method
will return the cell containing that point. The point can be specified
as a tuple, list or ``numpy`` array.

For example:

.. code-block:: python

  c = m.find((1200, 3450, -400))

finds the cell in the mesh ``m`` containing the point (1200, 3450,
-400) and stores it in the variable ``c`` (a ``cell`` object).

When used in this way, the ``find()`` method first determines the
layer containing the elevation of the point, and then searches that
layer for the appropriate column, using a quadtree search. If the mesh
does not contain the specified point, the ``find()`` method will
return ``None``.

Sometimes it may be more convenient to return the index of the cell,
rather than the cell object. This can be done by setting the
``indices`` parameter to ``True``:

.. code-block:: python

  i = m.find((1200, 3450, -400), indices = True)

In this case, ``i`` is an integer representing the cell
index. (However, ``None`` will still be returned if the point is not
inside the mesh.)

Searching using a function
..........................

It is also possible to use the ``find()`` method to search for cells
with particular attributes defined using a function. The function,
typically user-defined, must take a cell as its argument and return a
Boolean (``True`` or ``False``). The ``find()`` method will then
return a list of all mesh cells for which the function is ``True``.

For example, supposing we wish to find all mesh cells with volume
greater than 1000 and centre elevation below -600. To do this, we can
define a suitable function, and pass it to ``find()``:

.. code-block:: python

  def f(c):
    return c.volume > 1000 and c.layer.centre < -600

  cells = m.find(f)

Note that using a function to find cells in this way may not be
efficient for large meshes, as it involves a full search over all mesh
cells.

Searching for columns
---------------------

Searching by position
.....................

Passing a 2-D point (tuple, list or array) into the ``mesh``
``find()`` method will return the column containing that horizontal
position, for example:

.. code-block:: python

  p = np.array([3100, 4410])
  col = m.find(p)

returns the column object containing the point (3100, 4410)
(represented here by the ``numpy`` array ``p``) and stores it in the
variable ``col``. As for cells, setting the ``indices`` parameter to
``True`` means that column indices can be returned instead of column
objects. In either case, ``None`` is returned if the point is outside
the mesh.

Searching for columns inside a polygon
......................................

Passing a polygon of 2-D points into the ``mesh`` ``find()`` method
will return a list of all columns with centroids inside that polygon.

Here, a polygon is represented by a tuple, list or array of 2-D points
(each one a tuple, list or array of length 2). To search inside a
rectangle, only the bottom left and top right corner points need be
specified (any polygon with only two points will be interpreted as a
rectangle).

For example:

.. code-block:: python

  cols = m.find([(0,0), (3500, 4200)])

finds all columns in the rectangle with bottom left corner at the
origin and top right corner at (3500, 4200).

.. code-block:: python

  cols = m.find([(0,0), (3500, 1100), (900, 5100)])

finds all columns in the triangle with corners at the specified
points. Polygons may have any number of points.

Searching for layers
--------------------

Passing a scalar into the ``mesh`` ``find()`` method will return the
layer containing the specified elevation, e.g.:

.. code-block:: python

  lay = m.find(-2400)

returns the layer containing the elevation -2400. Again, the
``indices`` parameter can be used to return layer indices rather than
layer objects, and ``None`` is always returned if the elevation is
outside the mesh.

Searching within columns or layers
----------------------------------

Mesh columns and layers also have a ``find()`` method, which works
very similarly to that of the mesh itself. Passing a 3-D point as
parameter will return the cell containing that point (or ``None`` if
it is outside), e.g.:

.. code-block:: python

  c = m.layer[-1].find((3450, 1200, -340))

finds the cell in the bottom layer (index -1) of the mesh containing
the point (3450, 1200, -340).

.. code-block:: python

  c = m.column[12].find((3450, 1200, -340))

searches column 12 in the mesh for the same 3-D point.

Passing a 2-D point will return the column containing that point. If a
column is being searched, the result will be either the column itself,
or ``None``. For example:

.. code-block:: python

  col = m.layer[2].find((230, 345))

finds the column in mesh layer 2 containing the point (230, 345). Note
that the search results can be different in different layers, because
not all of them necessarily contain the same columns (if there are
incomplete layers at the surface).

.. code-block:: python

  if m.column[12].find((230, 345)):
    # do something

uses ``find()`` in a conditional to execute some code if the
horizontal point (230, 345) is inside column 12 of the mesh.

Passing a scalar will return the layer containing that elevation. If a
layer is being searched, either the layer itself or ``None`` will be
returned. For example:

.. code-block:: python

  lay = m.column[12].find(-100)

returns the layer in column 12 containing the elevation -100.

.. code-block:: python

  if m.layer[-1].find(-3000):
    # do something

executes a conditional statement if the bottom mesh layer contains the
elevation -3000.

Cells in columns and layers can also be found using a function, in
exactly the same way this is done for a mesh.

Searching within cells
----------------------

It is also possible to search within a cell. This amounts to
determining if the cell contains the specified 3-D point, 2-D
horizontal position or scalar elevation. If it does, the cell, column
or layer itself is returned. If it doesn't, ``None`` is returned.

For example:

.. code-block:: python

  if m.cell[2].find((230, 540, -250)):
    # do something

executes a conditional if cell 2 in the mesh contains the 3-D point
(230, 540, -250).

Passing a polygon into a cell's ``find()`` method will return the
cell's column if its centroid is inside the polygon (or ``None``
otherwise).
