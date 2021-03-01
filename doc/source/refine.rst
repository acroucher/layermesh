.. _refine:

Horizontal refinement
=====================

Horizontal mesh refinement (i.e. refinement of the column structure)
can be carried out using the ``mesh`` ``refine()`` method. It is
possible to refine only selected parts of the mesh using the
``columns`` parameter, which is a set, tuple or list of ``column``
objects to be refined.

For example:

.. code-block:: python

  cols = m.find((0, 0), (4000, 5000))
  m.refine(cols)

will refine all mesh columns within the rectangle with lower left
corner at the origin and upper right corner at (4000, 5000).

The selected columns are replaced by four refined columns (the edges
of the original columns being subdivided in two). Triangular columns
are added around the edge of the refinement area to make the
transition from coarse to fine columns.

Note that the triangular transition columns created by ``refine()``
may not necessarily have desirable mesh quality statistics
(e.g. aspect ratios or face orthogonality). Hence it is often
necessary to follow the ``refine()`` command with a call to the
``optimize()`` method (see :ref:`optimize`), in order to regain
acceptable mesh quality in the transition region.


