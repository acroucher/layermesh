Importing Layermesh in a Python script
======================================

Before you can use Layermesh in a Python script, it must be imported
(just like any other Python package). The package name is lowercase:
``layermesh``.

The ``layermesh`` package contains several modules, the most important
of which is the ``mesh`` module. There are several different ways a
module can be imported from a Python library. Perhaps the simplest is
to use the following syntax:

.. code-block:: python

  from layermesh import mesh

This imports only the ``mesh`` module from the ``layermesh`` package
(which is typically all that is needed - the other Layermesh modules
are just ones used by the ``mesh`` module). Commands from this module
must be prefixed by the module name, ``mesh``.

It is possible to import the module under a different name. For example:

.. code-block:: python

  from layermesh import mesh as lm

imports the mesh module and renames it to ``lm``. Then, mesh commands
would be prefixed by ``lm`` instead of ``mesh``.
