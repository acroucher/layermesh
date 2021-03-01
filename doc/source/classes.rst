.. index:: Layermesh; classes, classes

Layermesh classes
=================

Layermesh provides the following main Python classes for representing
meshes and mesh components.

* ``mesh``: class for a layer/column mesh
* ``node``: class for a 2-D horizontal mesh node
* ``column``: class for a mesh column, defined by a list of ``node`` objects
* ``layer``: class for a mesh layer, defined by its top and bottom elevations
* ``cell``: class for a mesh cell at a particular ``layer`` and ``column``

For full documentation of these classes, see the :ref:`layermeshapi`.

.. index:: classes; mesh

The ``mesh`` class
------------------

A ``mesh`` object represents an entire mesh.

It has list properties containing its nodes, columns, layers and
cells. These are called ``node``, ``column``, ``layer`` and ``cell``
respectively, and their elements are all objects of the appropriate
type.

.. index:: classes; node

The ``node`` class
------------------

A ``node`` object is defined mainly by its position property ``pos``,
a ``numpy`` array of length 2, representing its horizontal
location. It also has a ``column`` property, a set of the columns the
node belongs to.

.. index:: classes; column

The ``column`` class
--------------------

A ``column`` object is defined mainly by its nodes, which are stored
in its ``node`` property - a list of ``node`` objects. It also has a
``neighbour`` property, a set of the neighbouring columns (those which
share a face).

A ``column`` object also has ``layer`` and ``cell`` list properties,
containing the layers and cells in the column. Note that different
columns may have different numbers of layers, as the upper layers in
the mesh may be incomplete, to represent e.g. surface topography.

Columns also have geometric properties derived from their node
positions, e.g. ``area`` and ``centroid``, and a ``surface`` property,
which is the elevation of the top of the column.

.. index:: classes; layer

The ``layer`` class
-------------------

A ``layer`` object is defined mainly by its ``top`` and ``bottom``
properties, which are scalars representing the top and bottom
elevations of the layer.

A ``layer`` object also has ``column`` and ``cell`` list properties,
containing the columns and cells in the layer, as well as a
``column_cell`` property, for locating layer cells by their column
index. Note that different layers may have different numbers of
columns, as the upper layers may be incomplete.

Layers also have geometric properties derived from their top and
bottom elevations, e.g. ``centre`` and ``thickness``.

.. index:: classes; cell

The ``cell`` class
------------------

A ``cell`` object is defined by its ``layer`` and ``column``
properties, which are the layer and column objects corresponding to
the cell.

Cells have geometric properties such as ``volume`` and
``centroid``. Other useful properties can be accessed via the
``column`` and ``layer`` properties. For example, for a cell object
``c``, the horizontal area is given by ``c.column.area``, and its
vertical height is given by ``c.layer.thickness``.

Index properties
----------------

Instances of the ``node``, ``column``, ``layer`` and ``cell`` classes
all have an ``index`` property. This represents their index in the
corresponding list in the mesh they belong to.

For example, for a column ``col`` which is part of a mesh ``m``,
``col.index`` gives the index of ``col`` in the ``m.column`` list.

