Creating rectangular meshes
===========================

A simple rectangular mesh can be created by using the `rectangular`
parameter. This is a list or tuple of the mesh spacings in each
coordinate direction. Each mesh spacing specification is itself a
list, tuple or array of spacings.

For example:

.. code-block:: python

  from layermesh import mesh as lm
  m = lm.mesh(rectangular = ([1000]*10, [800]*12, [100]*10))

creates 
