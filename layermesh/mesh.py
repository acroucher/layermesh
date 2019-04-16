"""Layered computational meshes.

Copyright 2019 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

class node(object):
    """2-D mesh node."""

    def __init__(self, pos):
        self.pos = pos
        self.column = set()

    def __repr__(self):
        return str(list(self.pos))

class column(object):
    """Mesh column."""

    def __init__(self, node, surface = None):
        self.node = node
        self.surface = surface
        if self.num_nodes > 0:
            self.centre = self.centroid

    def __repr__(self):
        return str(self.node)

    def get_num_nodes(self): return len(self.node)
    num_nodes = property(get_num_nodes)

    def get_polygon(self):
        """Returns polygon formed by column node positions."""
        return [node.pos for node in self.node]
    polygon = property(get_polygon)

    def get_centroid(self):
        """Returns column centroid."""
        from geometry import polygon_centroid
        return polygon_centroid(self.polygon)
    centroid = property(get_centroid)

class layer(object):
    """Mesh layer."""

    def __init__(self, bottom, top):
        self.bottom = bottom
        self.top = top

    def __repr__(self):
        return str(self.bottom) + ': ' + str(self.top)

class mesh(object):
    """Layered computational mesh."""

    def __init__(self, filename = None, columns = None, layers = None):
        """Initialize layered mesh either from file or specified columns and
        layers."""

        if columns is not None:
            if isinstance(columns, (list, tuple)) and len(columns) == 2:
                self.set_rectangular_columns(columns)
            else:
                raise Exception("Unrecognized columns parameter.")

        if layers is not None:
            self.set_layers(layers)

    def get_num_nodes(self):
        return len(self.node)
    num_nodes = property(get_num_nodes)

    def get_num_columns(self):
        return len(self.column)
    num_columns = property(get_num_columns)

    def get_num_layers(self):
        return len(self.layer)
    num_layers = property(get_num_layers)

    def add_node(self, n):
        """Adds horizontal node to mesh."""
        self.node.append(n)

    def add_layer(self, lay):
        """Adds layer to mesh."""
        self.layer.append(lay)

    def add_column(self, col):
        """Adds column to mesh."""
        self.column.append(col)
        for n in col.node:
            n.column.add(col)

    def set_rectangular_columns(self, spacings):
        """Sets rectangular mesh columns according to specified horizontal
        mesh spacings."""

        self.node = []
        self.column = []
        verts = [np.concatenate([np.zeros(1), np.cumsum(np.array(s))])
                 for s in spacings]
        nv0 = len(verts[0])
        ns = [len(s) for s in spacings]
        for y in verts[1]:
            for x in verts[0]:
                n = node(pos = np.array([x, y]))
                self.add_node(n)
        for j in range(ns[1]):
            for i in range(ns[0]):
                column_node_indices = [
                    j * nv0 + i,
                    (j + 1) * nv0 + i,
                    (j + 1) * nv0 + i + 1,
                    j * nv0 + i + 1]
                column_nodes = [self.node[ind]
                                for ind in column_node_indices]
                col = column(node = column_nodes)
                self.add_column(col)

    def set_layers(self, spacings):
        """Sets mesh layers according to specified vertical layer spacings,
        from the top down."""

        self.layer = []
        z = 0.
        for thickness in spacings:
            top = z
            z -= thickness
            bottom = z
            lay = layer(bottom, top)
            self.add_layer(lay)
