.. index:: mesh; optimizing
.. _optimize:

Optimizing a mesh
=================

The accuracy of the results generated from a simulation on a
computational mesh is dependent (in part) on the quality of the
mesh. The way mesh quality is measured depends on the type of
simulation being carried out.

For some types of finite element simulation, for example, elements
should ideally have small aspect ratios and low skewness. For some
types of finite volume simulations, on the other hand, the
orthogonality of the mesh faces is important for accurate results.

Layermesh ``mesh`` objects have an ``optimize()`` method for improving
mesh quality. This method uses a least-squares minimization technique
to move the mesh node positions in such a way as to maximize specified
mesh quality measures.

Any weighted combination of aspect ratio, skewness and face
orthogonality may be used in the optimization. This is specified via
the ``weight`` parameter, which is a dictionary with up to three keys:
"aspect", "skewness" and "orthogonal". The values assigned to these
keys are the desired relative weights of the corresponding mesh
quality measures in the optimization.

In general it is not advisable to attempt to optimize the entire mesh
at once. This gives an optimization with too many degrees of freedom,
resulting in long processing times and a greater chance of either
non-convergence or convergence to a nonsensical result. It is better
to focus the optimization on those areas of the mesh that are known to
need improvement. For example, if the mesh has had horizontal local
refinement (see :ref:`refine`), the triangular transition columns may
have low quality, in which case the optimization can be concentrated
on those columns.

The optimization can be limited to either specified nodes, or
specified columns. In the latter case, all nodes in the specified
columns are selected for optimization.

For example:

.. code-block:: python

  triangles = m.type_columns(3)
  m.optimize(columns = triangles)

optimizes all nodes in triangular columns, using the default
weighting, which gives face orthogonality a weight of 1 and other
measures zero (i.e. only face orthogonality is optimized).

.. code-block:: python

  cols = m.find([(0, 0), (4000, 5000)])
  m.optimize(columns = cols, weight = {'aspect': 0.75, 'skewness': 0.25})

Here all nodes in columns within a specified rectangle are optimized,
giving 75% weight to aspect ratio and 25% to skewness in the
optimization. (Orthogonality is not specified, so it is given zero
weight.)
