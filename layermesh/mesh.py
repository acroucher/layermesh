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
        self.pos = np.array(pos)
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
        self._centroid = None
        self._area = None

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

    def get_centroid(self):
        """Returns column centroid."""
        if self._centroid is None:
            from layermesh.geometry import polygon_centroid
            self._centroid = polygon_centroid(self.polygon)
        return self._centroid
    centroid = property(get_centroid)
    centre = property(get_centroid)

    def get_area(self):
        """Returns column area."""
        if self._area is None:
            from layermesh.geometry import polygon_area
            self._area = polygon_area(self.polygon)
        return self._area
    area = property(get_area)

    def get_volume(self):
        """Returns column volume."""
        return self.area * sum([lay.thickness for lay in self.layer])
    volume = property(get_volume)

    def contains(self, pos):
        """Returns True if the column contains the 2-D point pos (tuple, list or
        numpy array of length 2)."""
        from layermesh.geometry import in_polygon
        return in_polygon(np.array(pos), self.polygon)

    def inside(self, polygon):
        """Returns true if the centre of the column is inside the specified
        polygon."""
        from layermesh.geometry import in_rectangle, in_polygon
        if len(polygon) == 2:
            return in_rectangle(self.centre, polygon)
        else:
            return in_polygon(self.centre, polygon)

    def translate(self, shift):
        """Translates column horizontally by the specified shift array."""
        if self._centroid is not None:
            self._centroid += np.array(shift)

class layer(object):
    """Mesh layer."""

    def __init__(self, bottom, top, index = None):
        self.bottom = bottom
        self.top = top
        self.index = index
        self._centre = None

    def __repr__(self):
        return str(self.index)

    def get_num_columns(self): return len(self.column)
    num_columns = property(get_num_columns)

    def get_num_cells(self): return len(self.cell)
    num_cells = property(get_num_cells)

    def get_centre(self):
        """Returns layer centre."""
        if self._centre is None:
            self._centre = 0.5 * (self.bottom + self.top)
        return self._centre
    centre = property(get_centre)

    def get_thickness(self):
        """Returns layer thickness."""
        return self.top - self.bottom
    thickness = property(get_thickness)

    def get_area(self):
        """Returns area of layer."""
        return sum([col.area for col in self.column])
    area = property(get_area)

    def get_volume(self):
        """Returns volume of layer."""
        return self.area * self.thickness
    volume = property(get_volume)

    def translate(self, shift):
        """Translates layer vertically by specified shift."""
        self.bottom += shift
        self.top += shift
        if self._centre is not None:
            self._centre += shift

    def contains(self, z):
        """Returns True if layer contains specified elevation z, or False
        otherwise."""
        return self.bottom <= z <= self.top

    def cell_containing(self, pos):
        """Returns cell in layer with column containing the 2-D point pos (a
        tuple, list or numpy array of length 2). If no column in the
        layer contains this point then None is returned.
        """

        for c in self.cell:
            if c.column.contains(pos): return c
        return None

    def cells_inside(self, polygon):
        """Returns a list of cells in the layer with columns inside the
        specified polygon."""
        return [c for c in self.cell if c.column.inside(polygon)]

    def find(self, match, indices = False):
        """Returns cell or cells in the layer satifying the specified matching
        criterion. The match parameter can be a function taking a cell
        and returning a Boolean, in which case a list of matching
        cells is returned. Alternatively it can be a 2-D point (tuple,
        list or numpy array), in which case the cell with column
        containing the point is returned, or a 2-D polygon, in which
        case a list of cells with columns inside the polygon is
        returned.  If indices is True, the cell indices are returned
        rather than the cells themselves.
        """

        if isinstance(match, (tuple, list)) and \
           all([isinstance(item, (float, int)) for item in match]) or \
           isinstance(match, np.ndarray):
            if len(match) == 2:
                c = self.cell_containing(match)
                if c is None: return None
                else: return c.index if indices else c
            else:
                raise Exception('Point to match is not of length 2.')
        elif isinstance(match, (list, tuple)) and \
             all([isinstance(item, (tuple, list, np.ndarray)) and \
                             len(item) == 2 for item in match]):
            cells = self.cells_inside(match)
            if cells:
                return [c.index for c in cells] if indices else cells
            else: return []
        elif callable(match):
            cells = [c for c in self.cell if match(c)]
            if cells:
                return [c.index for c in cells] if indices else cells
            else: return []
        else:
            raise Exception('Unrecognised match type.')

class cell(object):
    """Mesh cell."""

    def __init__(self, lay, col, index = None):
        self.layer = lay
        self.column = col
        self.index = index

    def __repr__(self):
        return str(self.index)

    def get_volume(self):
        """Returns cell volume."""
        return self.layer.thickness * self.column.area
    volume = property(get_volume)

    def get_centroid(self):
        """Returns cell centroid."""
        return np.concatenate([self.column.centroid,
                               np.array([self.layer.centre])])
    centroid = property(get_centroid)
    centre = property(get_centroid)

    def get_surface(self):
        """Returns True if cell is at surface of mesh, False otherwise."""
        return self == self.column.cell[0]
    surface = property(get_surface)

    def get_num_nodes(self):
        """Returns number of nodes in the cell."""
        return 2 * self.column.num_nodes
    num_nodes = property(get_num_nodes)

class mesh(object):
    """Layered computational mesh."""

    def __init__(self, filename = None, columns = None, layers = None,
                 surface = None, cells_type_sort = True):
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

        self.cells_type_sort = cells_type_sort
        self.setup()

    def __repr__(self):
        return '%d columns, %d layers, %d cells' % \
            (self.num_columns, self.num_layers, self.num_cells)

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

    def get_centre(self):
        """Returns horizontal centre of mesh, approximated by an area-weighted
        average of column centres."""
        if self.num_columns > 0:
            c = np.zeros(2)
            a = 0.
            for col in self.column:
                c += col.centre * col.area
                a += col.area
            return c / a
        else: return None
    centre = property(get_centre)

    def get_bounds(self):
        """Returns horizontal bounding box for mesh."""
        from layermesh.geometry import bounds_of_points
        return bounds_of_points([node.pos for node in self.node])
    bounds = property(get_bounds)

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
        if self.cells_type_sort:
            self._setup_cells_type_sorted()
        else:
            self._setup_cells_type_unsorted()

    def _setup_cells_type_sorted(self):
        """Sets up cells, sorted by cell type."""
        cells = {}
        for lay in self.layer:
            lay.cell = []
            for col in lay.column:
                c = cell(lay, col)
                cell_type = cell.num_nodes
                if cell_type not in cells: cells[cell_type] = []
                cells[cell_type].append(c)
                lay.cell.append(c)
                col.cell.append(c)

        cell_types = cells.keys()
        if self.cells_type_sort == 'increasing':
            cell_types = sorted(cell_types)
        elif self.cells_type_sort in [True, 'decreasing']:
            cell_types = sorted(cell_types, reverse = True)
        elif isinstance(self.cells_type_sort, [list, tuple, np.ndarray]):
            cell_types = self.cells_type_sort
        else:
            raise Exception('Unrecognised cell type sort: %s' % str(self.cells_type_sort))

        index = 0
        for cell_type in cell_types:
            if cell_type in cells:
                for c in cells[cell_type]:
                    c.index = index
                    self.cell.append(c)
                    index += 1

    def _setup_cells_type_unsorted(self):
        """Sets up cells, not sorted by cell type."""
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
                n = node(pos = [x, y], index = index)
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

        for c in self.cell:
            elt = []
            for iz, z in enumerate([c.layer.top, c.layer.bottom]):
                ilayer = c.layer.index + iz
                for n in c.column.node:
                    k = (ilayer, n.index)
                    if k not in node_index: # create point:
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

    def translate(self, shift):
        """Translates mesh by specified 3-D shift vector."""
        if isinstance(shift, (list, tuple)): shift = np.array(shift)
        for node in self.node: node.pos += shift[:2]
        for col in self.column:
            col.translate(shift[:2])
            if col.surface is not None:
                col.surface += shift[2]
        for layer in self.layer: layer.translate(shift[2])

    def rotate(self, angle, centre = None):
        """Rotates mesh horizontally by the specified angle (degrees
        clockwise). If no centre is specified, the mesh is rotated
        about its own centre."""
        from layermesh.geometry import rotation
        if centre is None:
            c = self.centre
        else:
            c = np.array(centre)
        A, b = rotation(angle, c)
        for n in self.node: n.pos = np.dot(A, n.pos) + b
        for col in self.column:
            if col._centroid is not None:
                col._centroid = np.dot(A, col._centroid) + b

    def find_layer(self, z):
        """Returns layer containing elevation z, or None if the point is
        outside the mesh."""
        if self.layer[-1].bottom <= z <= self.layer[0].top:
            i0, i1 = 0, self.num_layers - 1
            while i1 > i0:
                im = (i0 + i1) // 2
                if z >= self.layer[im].bottom: i1 = im
                else: i0 = im + 1
            return self.layer[i1]
        else:
            return None

    def find_column(self, pos):
        """Returns column containing point pos (list, tuple or numpy array of
        length 2), or None if pos is outside the mesh."""
        return self.layer[-1].find(pos).column

    def find_cell(self, pos):
        """Returns cell containing point pos (list, tuple or numpy array of
        length 3), or None if pos is outside the mesh."""

        lay = self.find_layer(pos[2])
        if lay is None:
            return None
        else:
            return lay.find(pos[:2])

    def find(self, match, indices = False):
        """Returns cell or cells matching the specified matching
        criterion. The match parameter can be a function taking a cell
        and returning a Boolean, in which case a list of matching
        cells is returned. Alternatively it can be a 3-D point (tuple,
        list or numpy array), in which case the cell containing the
        point is returned, or a 2-D point, in which case the column
        containing the point is returned, or a 1-D point or scalar,
        in which case the layer containing the elevation is returned.
        If indices is True, the cell (or column or layer) indices are
        returned rather than the cells, columns or layers themselves.
        """

        if isinstance(match, (tuple, list, np.ndarray)):
            if len(match) == 3:
                c = self.find_cell(match)
                if c is None: return None
                else: return c.index if indices else c
            elif len(match) == 2:
                c = self.find_column(match)
                if c is None: return None
                else: return c.index if indices else c
            elif len(match) == 1:
                l = self.find_layer(match[0])
                if l is None: return None
                else: return l.index if indices else l
            else:
                raise Exception('Point to match has length > 3.')
        elif isinstance(match, (float, int)):
            l = self.find_layer(match)
            if l is None: return None
            else: return l.index if indices else l
        elif callable(match):
            cells = [c for c in self.cell if match(c)]
            if cells:
                return [c.index for c in cells] if indices else cells
            else: return []
        else:
            raise Exception('Unrecognised match type.')

    def cells_inside(self, polygon, elevations = None, sort = True, indices = False):
        """Returns a list of cells in the mesh with columns inside the
        specified polygon. Specifying the elevations parameter as a two-element
        list, tuple or array means only cells inside that elevation range are returned.
        If sort is True, the returned cells are sorted by cell index. If indices is
        True, cell indices are returned instead of cells."""

        cols = [c.column for c in self.layer[-1].cells_inside(polygon)]
        cells = []
        if elevations is None:
            for col in cols:
                cells += col.cell
        else:
            for col in cols:
                cells += [c for c in col.cell
                          if elevations[0] <= c.layer.centre <= elevations[1]]
        if cells:
            if sort:
                isort = np.argsort(np.array([c.index for c in cells]))
                return [cells[i].index for i in isort] if indices \
                    else [cells[i] for i in isort]
            else:
                return [c.index for c in cells] if indices else cells
        else: return []

    def layer_plot(self, **kwargs):
        """Creates a 2-D Matplotlib plot of the mesh at a specified layer."""

        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if 'elevation' in kwargs:
            z = kwargs['elevation']
            lay = self.find(z)
            if lay is None:
                raise Exception('Elevation not found in layer_plot()')
        else:
            lay = kwargs.get('layer', self.layer[-1])
            if isinstance(lay, layer):
                if lay not in self.layer:
                    raise Exception('Unknown layer in layer_plot()')
            elif isinstance(lay, int):
                try:
                    lay = self.layer[lay]
                except:
                    raise Exception('Unknown layer in layer_plot()')


        linewidth = kwargs.get('linewidth', 0.2)
        linecolour = kwargs.get('linecolour', 'black')

        verts = []
        for col in lay.column:
            poslist = [tuple([p for p in n.pos])
                                for n in col.node]
            verts.append(tuple(poslist))

        polys = collections.PolyCollection(verts,
                                           linewidth = linewidth,
                                           facecolors = [],
                                           edgecolors = linecolour)

        if 'axes' in kwargs: ax = kwargs['axes']
        else: fig, ax = plt.subplots()

        ax.add_collection(polys)

        ax.set_aspect('equal')
        ax.autoscale_view()

        if 'axes' not in kwargs: plt.show()
