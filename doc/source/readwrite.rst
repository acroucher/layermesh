.. index:: mesh; reading, mesh; writing

Reading and writing HDF5 mesh files
===================================

A Layermesh ``mesh`` object can be written to an
`HDF5 file <https://www.hdfgroup.org/solutions/hdf5/>`_ using its
``write()`` method, which takes a filename as its parameter, e.g.:

.. code-block:: python

   msh.write('mymesh.h5')

writes the ``mesh`` object ``msh`` to the file "mymesh.h5".

Similarly, a ``mesh`` object can be read in from file by passing in a
filename when creating it:

.. code-block:: python

   import layermesh.mesh as lm
   msh = lm.mesh('mymesh.h5')

creates a new ``mesh`` object called ``msh`` and reads its contents
from the file "mymesh.h5".

.. index:: mesh; HDF files

Layermesh HDF5 files have a simple structure with four groups:

* ``cell``: one integer scalar ``type_sort`` dataset containing the
  value of the mesh ``cell_type_sort`` property
* ``layer``: one rank-1 array float ``elevation`` dataset containing,
  in order, the ``top`` property of each mesh layer, and finally the
  ``bottom`` property of the last (bottom) layer
* ``node``: one rank-2 array float ``position`` dataset containing the
  ``pos`` property of each node (horizontal position) in the mesh
  ``node`` list
* ``column``: an rank-2 integer ``node`` dataset containing the index
  of each node in the column, for each column in the mesh ``column``
  list; and also a rank-1 integer ``num_layers`` dataset containing the
  number of layers for each column


