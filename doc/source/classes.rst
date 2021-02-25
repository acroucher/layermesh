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

The ``mesh`` class
------------------

A ``mesh`` object represents an entire mesh.

It has list properties containing its nodes, columns, layers and
cells. These are called ``node``, ``column``, ``layer`` and ``cell``
respectively, and their elements are all objects of the appropriate
type.

The ``node`` class
------------------

A ``node`` is defined mainly by its position property ``pos``, a
``numpy`` array of length 2, representing its horizontal location.

.. index properties

