"""Layered computational meshes.

Copyright 2019 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

def memoize(f):
    """Decorator for caching function values."""
    memo = {}
    def check(x):
        if x not in memo: memo[x] = f(x)
        return memo[x]
    return check

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

    def __init__(self, node, index = None, surface = None):
        self.node = node
        self.index = index
        self.surface = surface

    def __repr__(self):
        return str(self.index)

    def get_num_nodes(self): return len(self.node)
    num_nodes = property(get_num_nodes)

    def get_num_layers(self): return len(self.layer)
    num_layers = property(get_num_layers)

    def get_num_cells(self): return len(self.cell)
    num_cells = property(get_num_cells)

    def get_polygon(self):
        """Returns polygon formed by column node positions."""
        return [node.pos for node in self.node]
    polygon = property(get_polygon)

    @memoize
    def get_centroid(self):
        """Returns column centroid."""
        from geometry import polygon_centroid
        return polygon_centroid(self.polygon)
    centroid = property(get_centroid)
    centre = property(get_centroid)

    @memoize
    def get_area(self):
        """Returns column area."""
        from geometry import polygon_area
        return polygon_area(self.polygon)
    area = property(get_area)

    @memoize
    def get_volume(self):
        """Returns column volume."""
        return self.area * sum([lay.thickness for lay in self.layer])
    volume = property(get_volume)

class layer(object):
    """Mesh layer."""

    def __init__(self, bottom, top, index = None):
        self.bottom = bottom
        self.top = top
        self.index = index

    def __repr__(self):
        return str(self.index)

    def get_num_columns(self): return len(self.column)
    num_columns = property(get_num_columns)

    def get_num_cells(self): return len(self.cell)
    num_cells = property(get_num_cells)

    @memoize
    def get_centre(self):
        """Returns layer centre."""
        return 0.5 * (self.bottom + self.top)
    centre = property(get_centre)

    @memoize
    def get_thickness(self):
        """Returns layer thickness."""
        return self.top - self.bottom
    thickness = property(get_thickness)

class cell(object):
    """Mesh cell."""

    def __init__(self, lay, col, index = None):
        self.layer = lay
        self.column = col
        self.index = index

    def __repr__(self):
        return str(self.index)

    @memoize
    def get_volume(self):
        """Returns cell volume."""
        return self.layer.thickness * self.column.area
    volume = property(get_volume)

    @memoize
    def get_centroid(self):
        """Returns cell centroid."""
        return np.concatenate([self.column.centroid,
                               np.array([self.layer.centre])])
    centroid = property(get_centroid)
    centre = property(get_centroid)

class mesh(object):
    """Layered computational mesh."""

    def __init__(self, filename = None, columns = None, layers = None,
                 surface = None):
        """Initialize layered mesh either from file or specified columns and
        layers."""

        if columns is not None:
            if isinstance(columns, (list, tuple)) and len(columns) == 2:
                self.set_rectangular_columns(columns)
            else:
                raise Exception("Unrecognized columns parameter.")

        if layers is not None:
            self.set_layers(layers)

        if surface is not None:
            if isinstance(surface, (dict, list, np.ndarray)):
                self.set_column_surfaces(surface)

        self.setup()

    def get_num_nodes(self):
        """Returns number of 2-D nodes in mesh."""
        return len(self.node)
    num_nodes = property(get_num_nodes)

    def get_num_columns(self):
        """Returns number of columns in mesh."""
        return len(self.column)
    num_columns = property(get_num_columns)

    def get_num_layers(self):
        """Returns number of layers in mesh."""
        return len(self.layer)
    num_layers = property(get_num_layers)

    def get_area(self):
        """Returns horizontal area of mesh. """
        return sum([col.area for col in self.column])
    area = property(get_area)

    def get_volume(self):
        """Returns total volume of mesh."""
        return sum([col.volume for col in self.column])
    volume = property(get_volume)

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

    def set_column_layers(self, col):
        """Populates list of layers for given column."""
        s = col.surface if col.surface is not None else self.layer[0].top
        col.layer = [lay for lay in self.layer if s >= lay.centre]

    def set_layer_columns(self, lay):
        """Populates list of columns for given layer."""
        lay.column = [col for col in self.column
                      if self.column_in_layer(col, lay)]

    def setup(self):
        """Sets up internal mesh variables."""
        for col in self.column:
            self.set_column_layers(col)
        for ilayer, lay in enumerate(self.layer):
            self.set_layer_columns(lay)
        self.setup_cells()

    def setup_cells(self):
        """Sets up cell properties of mesh, layers and columns."""
        self.cell = []
        for col in self.column: col.cell = []
        index = 0
        for lay in self.layer:
            lay.cell = []
            for col in lay.column:
                c = cell(lay, col, index)
                self.cell.append(c)
                lay.cell.append(c)
                col.cell.append(c)
                index += 1

    def get_num_cells(self):
        """Returns number of 3-D cells in mesh."""
        return len(self.cell)
    num_cells = property(get_num_cells)

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
        index = 0
        for j in range(ns[1]):
            for i in range(ns[0]):
                column_node_indices = [
                    j * nv0 + i,
                    (j + 1) * nv0 + i,
                    (j + 1) * nv0 + i + 1,
                    j * nv0 + i + 1]
                column_nodes = [self.node[ind]
                                for ind in column_node_indices]
                col = column(node = column_nodes, index = index)
                self.add_column(col)
                index += 1

    def set_layers(self, spacings):
        """Sets mesh layers according to specified vertical layer spacings,
        from the top down."""

        self.layer = []
        z = 0.
        index = 0
        for thickness in spacings:
            top = z
            z -= thickness
            bottom = z
            lay = layer(bottom, top, index)
            self.add_layer(lay)
            index += 1

    def column_in_layer(self, col, lay):
        """Returns true if column is in the specified layer, or
        false otherwise."""
        return col.num_layers >= self.num_layers - lay.index

    def set_column_surfaces(self, surface):
        """Sets column surface properties from surface dictionary (keyed by
        column indices) or list/array of values for all columns."""

        if isinstance(surface, dict):
            for icol, s in surface.items():
                self.column[icol].surface = s
        elif isinstance(surface, (list, np.ndarray)):
             if len(surface) == self.num_columns:
                 for col, s in zip(self.column, surface):
                     col.surface = s
             else:
                 raise Exception('Surface list or array is the wrong size.')
        else:
            raise Exception('Unrecognized surface parameter type.')

    def get_meshio_points_cells(self):
        """Returns lists of 3-D points and cells suitable for mesh
        input/output using meshio library."""

        points = []
        cells = {'wedge': [], 'hexahedron': []}
        cell_type = {6: 'wedge', 8: 'hexahedron'}
        node_index = {}
        point_index = 0

        for ilayer, lay in enumerate(self.layer):
            for col in lay.column:
                elt = []
                for i in [ilayer - 1, ilayer]:
                    for n in col.node:
                        k = (i, n.index)
                        if k not in node_index: # create point:
                            z = self.layer[i].bottom if i >= 0 else self.layer[0].top
                            pos = np.concatenate((n.pos, np.array([z])))
                            points.append(pos)
                            node_index[k] = point_index
                            point_index += 1
                        elt.append(node_index[k])
                cells[cell_type[len(elt)]].append(elt)

        points = np.array(points)
        cells = dict([(k, np.array(v)) for k, v in cells.items() if v])
        return points, cells

    meshio_points_cells = property(get_meshio_points_cells)

    def get_surface_cells(self):
        """Returns cells at mesh surface."""
        return [col.cell[0] for col in self.column]
    surface_cells = property(get_surface_cells)
