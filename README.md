![Unit tests](https://github.com/acroucher/layermesh/workflows/Unit%20tests/badge.svg) [![Documentation Status](https://readthedocs.org/projects/layermesh/badge/?version=latest)](https://layermesh.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/layermesh.svg)](https://badge.fury.io/py/layermesh)

# Layermesh

Layermesh is a Python library for creating and manipulating computational meshes with a layer/column structure, i.e. a (possibly unstructured) 2-D mesh projected down through a series of layers of constant thickness.

The uppermost layers of the mesh may be incomplete (i.e. do not contain cells for all columns), so that an irregular top surface can be used to represent e.g. topography.

The Layermesh library can be used to carry out a variety of actions on such meshes, including:

* creating meshes
* loading and saving from [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files
* exporting to a variety of 3-D mesh formats (via the
  [meshio](https://pypi.org/project/meshio/) library)
* fitting surface elevation data
* local refinement of the horizontal mesh
* optimization to improve horizontal mesh quality
* mesh searching, to locate particular cells, columns or layers
* 2-D layer and vertical slice plots (via [Matplotlib](https://matplotlib.org/))

## Documentation

Documentation for Layermesh can be found on [Read The Docs](https://layermesh.readthedocs.io/en/latest/).

## Installation

Layermesh can be installed via `pip`, Python's package manager:

```python
pip install layermesh
```

or if you don't have permissions for installing system-wide Python packages, you can just install it locally inside your own user account:

```python
pip install --user layermesh
```

This will download and install Layermesh from the Python Package Index ([PyPI](https://pypi.org)).

## Licensing

Layermesh is open-source software, released under the GNU [Lesser General Public License](https://www.gnu.org/licenses/lgpl-3.0.en.html) (LGPL) version 3.
