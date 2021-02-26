Exporting to other formats
==========================

The purpose of using Layermesh is usually to create a computational
mesh which can be used by other software (such as a flow simulator or
3-D visualisation package). This involves expanding the layer/column
structure of a Layermesh mesh into a full 3-D mesh, which can then be
exported to a mesh format which other software can read.

This can be done by using the ``mesh`` ``export()`` method, which
takes a filename as its parameter. The `meshio
<https://pypi.org/project/meshio/>`_ library is used to write the
mesh, so the mesh can be exported to any mesh format that ``meshio``
understands (ExodusII, GMSH, VTU, XDMF, H5M and more). The desired
format is determined from the filename extension. (Alternatively, it
can be explicitly specified using the ``fmt`` parameter.)

For example:

.. code-block:: python

   from layermesh import mesh as lm
   m = lm.mesh('mymesh.h5')
   m.export('mymesh.vtu')
   m.export('mymesh.msh')

reads a mesh from a Layermesh HDF5 file and exports it twice, first to
a VTU file for 3-D visualisation using e.g. `Paraview
<https://www.paraview.org/>`_, and then to GMSH ``*.msh`` format.


