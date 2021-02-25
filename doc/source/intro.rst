************
Introduction
************

What is Layermesh?
==================

Layermesh is a Python library for creating and manipulating
computational meshes with a layer/column structure, i.e. a (possibly
unstructured) 2-D mesh projected down through a series of layers of
constant thickness.

The uppermost layers of the mesh may be incomplete (i.e. do not
contain cells for all columns), so that an irregular top surface can
be used to represent e.g. topography.

The Layermesh library can be used to carry out a variety of actions on
such meshes, including:

* creating meshes
* loading and saving from `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ files
* exporting to a variety of 3-D mesh formats (via the `meshio
  <https://pypi.org/project/meshio/>`_ library)
* fitting surface elevation data
* local refinement of the horizontal mesh
* optimization to improve horizontal mesh quality
* mesh searching, to locate particular cells, columns or layers
* 2-D layer and vertical slice plots (via `Matplotlib <https://matplotlib.org/>`_)

Installation
============

Layermesh can be installed via ``pip``, Python's package manager:

.. code-block:: bash

   pip install layermesh

or if you don't have permissions for installing system-wide Python
packages, you can just install it locally inside your own user
account:

.. code-block:: bash

   pip install --user layermesh

This will download and install Layermesh from the Python Package Index
(`PyPI <https://pypi.org>`_).

Dependencies
============

Layermesh depends on several other Python libraries:

* ``numpy``: `Numerical Python <https://numpy.org/>`_
* ``scipy``: `Scientific Python <https://www.scipy.org/>`_
* ``h5py``: Python `interface <https://www.h5py.org/>`_ for HDF5
* ``meshio``: Python library for `mesh file input/output
  <https://pypi.org/project/meshio/>`_
* ``matplotlib``: Python `plotting library <https://matplotlib.org/>`_

These will be installed automatically if not already present, if
``pip`` is used as above to install Layermesh.

Licensing
=========

Layermesh is open-source software, released under the GNU `Lesser
General Public License
<https://www.gnu.org/licenses/lgpl-3.0.en.html>`_ (LGPL) version 3.

