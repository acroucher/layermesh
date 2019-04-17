"""Layered computational meshes.

Copyright 2019 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

class node(object):
    """2-D mesh node."""

    def __init__(self, pos, index = None):
        self.pos = pos
        self.index = index
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

    def get_centre(self):
        """Returns layer centre."""
        return 0.5 * (self.bottom + self.top)
    centre = property(get_centre)

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

        self.setup()

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

    def set_column_num_layers(self, col):
        """Sets number of layers in column, based on its surface value and the
        mesh layer structure."""
        s = col.surface if col.surface is not None else self.layer[0].top
        col.num_layers = len([lay for lay in self.layer if s >= lay.centre])

    def set_layer_columns(self, ilayer, lay):
        """Populates list of columns for given layer."""
        lay.column = []
        for col in self.column:
            if self.column_in_layer(col, ilayer):
                lay.column.append(col)

    def setup(self):
        """Sets up internal mesh variables."""
        for col in self.column:
            self.set_column_num_layers(col)
        for ilayer, lay in enumerate(self.layer):
            self.set_layer_columns(ilayer, lay)

    def set_rectangular_columns(self, spacings):
        """Sets rectangular mesh columns according to specified horizontal
        mesh spacings."""

        self.node = []
        self.column = []
        verts = [np.concatenate([np.zeros(1), np.cumsum(np.array(s))])
                 for s in spacings]
        nv0 = len(verts[0])
        ns = [len(s) for s in spacings]
        index = 0
        for y in verts[1]:
            for x in verts[0]:
                n = node(pos = np.array([x, y]), index = index)
                self.add_node(n); index += 1
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

    def column_in_layer(self, col, ilayer):
        """Returns true if column is in the layer with specified index, or
        false otherwise."""
        return col.num_layers >= self.num_layers - ilayer
