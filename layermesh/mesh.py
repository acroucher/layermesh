"""Layered computational meshes.

Copyright 2019 University of Auckland.

This file is part of layermesh.

layermesh is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

layermesh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with layermesh.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

default_cell_type_sort = -1 # decreasing

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

    def __init__(self, node, index = None):
        self.node = node
        self.index = index
        self._centroid = None
        self._area = None
        self.neighbour = set()

    def __repr__(self):
        return str(self.index)

    def get_num_nodes(self): return len(self.node)
    num_nodes = property(get_num_nodes)

    def get_num_layers(self): return len(self.layer)
    num_layers = property(get_num_layers)

    def get_num_cells(self): return len(self.cell)
    num_cells = property(get_num_cells)

    def get_num_neighbours(self): return len(self.neighbour)
    num_neighbours = property(get_num_neighbours)

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

    def get_bounding_box(self):
        """Returns column horizontal bounding box."""
        from layermesh.geometry import bounds_of_points
        return bounds_of_points([n.pos for n in self.node])
    bounding_box = property(get_bounding_box)

    def set_layers(self, layers, num_layers):
        """Sets column layers to be the last num_layers layers from the
        specified list."""
        istart = len(layers) - num_layers
        self.layer = layers[istart:]

    def set_surface(self, layers, surface = None):
        """Sets column layers from specified surface elevation."""
        if surface is None: self.layer = layers
        else:
            self.layer = [lay for lay in layers
                          if lay.centre <= surface]

    def get_surface(self):
        """Returns surface elevation of column."""
        return self.layer[0].top
    surface = property(get_surface)

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

    def identify_neighbours(self):
        """Identifies neighbouring columns, i.e. those sharing two nodes."""
        shared = {}
        col_dict = {}
        for n in self.node:
            for col in n.column:
                if col.index in shared: shared[col.index] += 1
                else:
                    col_dict[col.index] = col
                    shared[col.index] = 1
        del shared[self.index]
        self.neighbour = set([col_dict[index] for index in shared
                              if shared[index] == 2])
        for col in self.neighbour: col.neighbour.add(self)

    def get_side_neighbours(self):
        """Returns a list of neighbouring columns corresponding to each column
        side (None if the column side is on a boundary)."""
        nbrs = []
        for i, nodei in enumerate(self.node):
            i1 = (i + 1) % self.num_nodes
            side_nodes = set([nodei, self.node[i1]])
            nbr = None
            for col in self.neighbour:
                if side_nodes < set(col.node):
                    nbr = col
                    break
            nbrs.append(nbr)
        return nbrs
    side_neighbours = property(get_side_neighbours)

class layer(object):
    """Mesh layer."""

    def __init__(self, bottom, top, index = None):
        self.bottom = bottom
        self.top = top
        self.index = index
        self._centre = None
        self.quadtree = None

    def __repr__(self):
        return str(self.index)

    def get_num_columns(self): return len(self.column)
    num_columns = property(get_num_columns)

    def get_num_cells(self): return len(self.cell)
    num_cells = property(get_num_cells)

    def get_nodes(self):
        """Returns set of nodes in layer."""
        nodes = set()
        for col in self.column:
            for n in col.node: nodes.add(n)
        return nodes
    node = property(get_nodes)

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

    def get_horizontal_bounds(self):
        """Returns horizontal bounding box for layer."""
        from layermesh.geometry import bounds_of_points
        return bounds_of_points([n.pos for n in self.node])
    horizontal_bounds = property(get_horizontal_bounds)

    def setup_quadtree(self):
        """Sets up quadtree for column searching."""
        from layermesh import quadtree
        self.quadtree = quadtree.quadtree(self.horizontal_bounds,
                                          self.column)

    def translate(self, shift):
        """Translates layer by specified 3-D shift vector."""
        self.bottom += shift[2]
        self.top += shift[2]
        if self._centre is not None:
            self._centre += shift[2]
        if self.quadtree is not None:
            self.quadtree.translate(shift[:2])

    def contains(self, z):
        """Returns True if layer contains specified elevation z, or False
        otherwise."""
        return self.bottom <= z <= self.top

    def cell_containing(self, pos):
        """Returns cell in layer with column containing the 2-D point pos (a
        tuple, list or numpy array of length 2). If no column in the
        layer contains this point then None is returned.
        """

        if self.quadtree is None: self.setup_quadtree()
        col = self.quadtree.search(pos)
        if col:
            return self.column_cell[col.index]
        else: return None

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

    def __init__(self, filename = None, **kwargs):

        """Initialize layered mesh either from file or via other
        parameters."""

        if filename is not None: self.read(filename)
        else:
            self.empty()
            self.cell_type_sort = kwargs.get('cell_type_sort',
                                             default_cell_type_sort)
            rectangular = kwargs.get('rectangular', None)
            if rectangular is not None:
                thicknesses = rectangular[2]
                elevations = np.hstack((np.zeros(1),
                                       -np.cumsum(np.array(thicknesses))))
                self.set_layers(elevations)
                spacings = rectangular[:2]
                self.set_rectangular_columns(spacings)
            self.surface = kwargs.get('surface', None)

    def __repr__(self):
        return '%d columns, %d layers, %d cells' % \
            (self.num_columns, self.num_layers, self.num_cells)

    def empty(self):
        """Empties data arrays."""
        self.cell_type_sort = default_cell_type_sort
        self.node = []
        self.column = []
        self.layer = []
        self.cell = []

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
        col.identify_neighbours()

    def set_layer_columns(self, lay):
        """Populates list of columns for given layer."""
        lay.column = [col for col in self.column
                      if self.column_in_layer(col, lay)]

    def setup(self):
        """Sets up internal mesh variables."""
        for lay in self.layer:
            self.set_layer_columns(lay)
        self.setup_cells()

    def setup_cells(self):
        """Sets up cell properties of mesh, layers and columns."""
        self.cell = []
        for col in self.column: col.cell = []
        if self.cell_type_sort:
            self._setup_cells_type_sorted()
        else:
            self._setup_cells_type_unsorted()

    def _setup_cells_type_sorted(self):
        """Sets up cells, sorted by cell type."""
        cells = {}
        for lay in self.layer:
            lay.cell = []
            lay.column_cell = {}
            for col in lay.column:
                c = cell(lay, col)
                cell_type = cell.num_nodes
                if cell_type not in cells: cells[cell_type] = []
                cells[cell_type].append(c)
                lay.cell.append(c)
                lay.column_cell[col.index] = c
                col.cell.append(c)

        cell_types = cells.keys()
        if self.cell_type_sort > 0:
            cell_types = sorted(cell_types)
        elif self.cell_type_sort < 0:
            cell_types = sorted(cell_types, reverse = True)
        else:
            raise Exception('Unrecognised cell type sort: %s' % str(self.cell_type_sort))

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
            lay.column_cell = {}
            for col in lay.column:
                c = cell(lay, col, index)
                self.cell.append(c)
                lay.cell.append(c)
                lay.column_cell[col.index] = c
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

    def set_layers(self, elevations):
        """Sets mesh layers according to specified vertical layer boundary
        elevations, from the top down. """

        self.layer = []
        index = 0
        bottom = elevations[0]
        for z in elevations[1:]:
            top = bottom
            bottom = z
            lay = layer(bottom, top, index)
            self.add_layer(lay)
            index += 1

    def column_in_layer(self, col, lay):
        """Returns true if column is in the specified layer, or
        false otherwise."""
        return col.num_layers >= self.num_layers - lay.index

    def set_surface(self, surface):
        """Sets column layers from surface dictionary (keyed by
        column indices) or list/array of values for all columns."""

        if surface is None:
           for col in self.column: col.set_surface(self.layer)
        elif isinstance(surface, dict):
            for col in self.column:
                if col.index in surface:
                    col.set_surface(self.layer, surface[col.index])
                else:
                    col.set_surface(self.layer)
        elif isinstance(surface, (list, np.ndarray)):
             if len(surface) == self.num_columns:
                 for col, s in zip(self.column, surface):
                     col.set_surface(self.layer, s)
             else:
                 raise Exception('surface is the wrong size.')
        else:
            raise Exception('Unrecognized surface parameter type.')
        self.setup()

    def get_surface(self):
        """Returns array of column surface elevations."""
        return np.array([col.surface for col in self.column])

    surface = property(get_surface, set_surface)

    def set_column_layers(self, num_layers):
        """Sets column layers from dictionary (keyed by column indices) or
        list/array of layer counts for each column."""
        if num_layers is None:
           for col in self.column: col.set_surface(self.layer)
        elif isinstance(num_layers, dict):
            for col in self.column:
                if col.index in num_layers:
                    col.set_layers(self.layer, num_layers[col.index])
                else:
                    col.set_layers(self.layer)
        elif isinstance(num_layers, (list, np.ndarray)):
             if len(num_layers) == self.num_columns:
                 for col, n in zip(self.column, num_layers):
                     col.set_layers(self.layer, n)
             else:
                 raise Exception('num_layers is the wrong size.')
        else:
            raise Exception('Unrecognized num_layers parameter type.')
        self.setup()

    def write(self, filename):
        """Writes mesh to HDF5 file."""
        import h5py
        with h5py.File(filename, 'w') as f:
            cell_group = f.create_group('cell')
            cell_group.create_dataset('type_sort', data = self.cell_type_sort)
            if self.layer:
                layer_elev = [self.layer[0].top] + \
                         [lay.bottom for lay in self.layer]
                lay_group = f.create_group('layer')
                dset = lay_group.create_dataset('elevation', data = layer_elev)
                dset.attrs['description'] = 'Layer boundary elevations, ' + \
                                            'from top to bottom'
            if self.node:
                pos = np.array([n.pos for n in self.node])
                node_group = f.create_group('node')
                dset = node_group.create_dataset('position', data = pos)
                dset.attrs['description'] = 'Position of each node'
                if self.column:
                    col_node_indices = np.array([[n.index for n in col.node]
                                                 for col in self.column])
                    col_group = f.create_group('column')
                    dset = col_group.create_dataset('node', data = col_node_indices)
                    dset.attrs['description'] = 'Indices of nodes in each column'
                    num_layers = np.array([col.num_layers for col in self.column])
                    dset = col_group.create_dataset('num_layers', data = num_layers)
                    dset.attrs['description'] = 'Number of layers in each column'

    def read(self, filename):
        """Reads mesh from HDF5 file."""
        import h5py
        self.empty()
        num_layers = None
        with h5py.File(filename, 'r') as f:
            if 'cell' in f:
                cell_group = f['cell']
                if 'type_sort' in cell_group:
                    self.cell_type_sort = int(np.array(cell_group['type_sort']))
            if 'layer' in f:
                lay_group = f['layer']
                self.set_layers(np.array(lay_group['elevation']))
            if 'node' in f:
                node_group = f['node']
                if 'position' in node_group:
                    index = 0
                    for p in np.array(node_group['position']):
                        n = node(pos = p, index = index)
                        self.add_node(n); index += 1
                    index = 0
                    if 'column' in f:
                        col_group = f['column']
                        if 'node' in col_group:
                            for col_node_indices in np.array(col_group['node']):
                                col_nodes = [self.node[i]
                                             for i in col_node_indices]
                                col = column(node = col_nodes, index = index)
                                self.add_column(col); index += 1
                            if 'num_layers' in col_group:
                                num_layers = np.array(col_group['num_layers'])
        self.set_column_layers(num_layers)

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

    def export(self, filename, fmt = None):
        """Exports 3-D mesh using meshio, to file with the specified name. If
        the format is not specified via the fmt parameter, it is determined
        from the filename extension."""
        import meshio
        points, cells = self.meshio_points_cells
        meshio.write_points_cells(filename, points, cells, file_format = fmt)

    def get_surface_cells(self):
        """Returns cells at mesh surface."""
        return [col.cell[0] for col in self.column]
    surface_cells = property(get_surface_cells)

    def column_faces(self, columns = None):
        """Returns a set of mesh column faces, each one being a frozenset of
        the two column indices on either side of a face. A list of the
        columns may be optionally specified, otherwise all columns
        will be included.
        """
        if columns is None: columns = self.column
        col_dict = {col.index: col for col in columns}
        faces = set()
        for col in columns:
            for nbr in col.neighbour:
                if nbr.index in col_dict:
                    faces.add(frozenset([col.index, nbr.index]))
        return faces

    def translate(self, shift):
        """Translates mesh by specified 3-D shift vector."""
        if isinstance(shift, (list, tuple)): shift = np.array(shift)
        for node in self.node: node.pos += shift[:2]
        for col in self.column: col.translate(shift[:2])
        for layer in self.layer: layer.translate(shift)

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
        for lay in self.layer:
            if lay.quadtree is not None:
                lay.setup_quadtree()

    def find_layer(self, z):
        """Returns layer containing elevation z, or None if the point is
        outside the mesh."""
        if self.num_layers == 0:
            return None
        else:
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
        if self.num_layers == 0:
            return None
        else:
            c = self.layer[-1].find(pos)
            return c if c is None else c.column

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

    def column_track(self, line):
        """Returns a list of tuples of (column, entry_point, exit_point)
        representing the horizontal track traversed by the specified
        line through the grid.  Line is a tuple of two 2D arrays.  The
        resulting list is ordered by distance from the start of the
        line. Adapted from the PyTOUGH mulgrid column_track()
        method."""

        def furthest_intersection(poly, line):
            """Returns furthest intersection point between line and poly."""
            from layermesh.geometry import line_polygon_intersections
            pts, inds = line_polygon_intersections(poly, line,
                                                   bound_line = (True, False),
                                                   indices = True)
            if pts:
                d = np.array([np.linalg.norm(intpt - line[0]) for intpt in pts])
                i = np.argmax(d)
                return pts[i], inds[i]
            else: return None, None

        def find_track_start(line):
            """Finds starting point for track- an arbitrary point on the line that
            is inside the grid.  If the start point of the line is
            inside the grid, that is used; otherwise, a recursive
            bisection technique is used to find a point."""

            col, start_type = None, None
            for endpt, name in zip(line, ['start', 'end']):
                pos, col, start_type = endpt, self.find(endpt), name
                if col: break
            if not col: # line ends are both outside the grid:
                start_type = 'mid'
                max_levels = 7

                def find_start(line, level = 0):
                    midpt = 0.5 * (line[0] + line[1])
                    col = self.find(midpt)
                    if col: return midpt, col
                    else:
                        if level <= max_levels:
                            line0, line1 = [line[0], midpt], [midpt, line[1]]
                            pos, col = find_start(line0, level + 1)
                            if col: return pos, col
                            else:
                                pos, col = find_start(line1, level + 1)
                                if col: return pos, col
                                else: return None, None
                        else: return None, None

                pos, col = find_start(line)
            return pos, col, start_type

        def next_corner_column(col, pos, more, cols):
            """If the line has hit a node, determine a new column containing that
            node, not already visited."""

            node_tol = 1.e-12
            nextcol = None
            nearnodes = [n for n in col.node if np.linalg.norm(n.pos - pos) < node_tol]
            if nearnodes: # hit a node
                nearnode = nearnodes[0]
                nearcols = nearnode.column - cols
                if nearcols: nextcol = nearcols.pop()
                else: more = False
            return nextcol, more

        def next_neighbour_column(col, more, cols):
            """Determine a new neighbour column not already visited."""

            nbrs = col.neighbour - cols
            if nbrs: return nbrs.pop(), more
            else: return None, False

        def find_track_segment(linesegment, pos, col):
            """Finds track segment starting from the specified position and
            column."""
            track = []
            cols, more, inpos = set(), True, pos
            colnbr, nextcol = col.side_neighbours, None
            lined = np.linalg.norm(linesegment[1] - linesegment[0])
            while more:
                cols.add(col)
                outpos, ind = furthest_intersection(col.polygon, linesegment)
                if outpos is not None:
                    d = np.linalg.norm(outpos - linesegment[0])
                    if d >= lined: # gone past end of line
                        outpos = linesegment[1]
                        more = False
                    if np.linalg.norm(outpos - inpos) > 0.:
                        track.append(tuple([col, inpos, outpos]))
                    if more: # find next column
                        inpos = outpos
                        nextcol = colnbr[ind]
                        if nextcol:
                            if nextcol in cols:
                                nextcol, more = next_corner_column(col, outpos, more, cols)
                                if nextcol is None:
                                    nextcol, more = next_neighbour_column(col, more, cols)
                                    nbr_base_col = col
                        else: nextcol, more = next_corner_column(col, outpos, more, cols)
                else:
                    nextcol, more = next_neighbour_column(nbr_base_col, more, cols)
                col = nextcol
                if col: colnbr = col.side_neighbours
                else: more = False

            return track

        def reverse_track(track):
            return [tuple([tk[0], tk[2], tk[1]]) for tk in track][::-1]

        line = [np.array(p) for p in line]
        pos, col, start_type = find_track_start(line)
        if pos is not None and col:
            if start_type == 'start':
                track = find_track_segment(line, pos, col)
            elif start_type == 'end':
                track = find_track_segment(line[::-1], pos, col)
                track = reverse_track(track)
            else:
                track1 = find_track_segment([pos, line[0]], pos, col)
                track2 = find_track_segment([pos, line[1]], pos, col)
                # remove arbitrary starting point from middle of track, and join:
                midtk = tuple([track1[0][0], track1[0][2], track2[0][2]])
                track = reverse_track(track1)[:-1] + [midtk] + track2[1:]
            return track
        else: return []

    def layer_plot(self, lay = -1, **kwargs):
        """Creates a 2-D Matplotlib plot of the mesh at a specified layer."""

        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if 'axes' in kwargs: ax = kwargs['axes']
        else: fig, ax = plt.subplots()

        if 'elevation' in kwargs:
            z = kwargs['elevation']
            lay = self.find(z)
            if lay is None:
                raise Exception('Elevation not found in layer_plot()')
        else:
            if isinstance(lay, layer):
                if lay not in self.layer:
                    raise Exception('Unknown layer in layer_plot()')
            elif isinstance(lay, int):
                try:
                    lay = self.layer[lay]
                except:
                    raise Exception('Unknown layer in layer_plot()')

        labels = kwargs.get('label', None)
        label_fmt = kwargs.get('label_format', '%g')
        label_colour = kwargs.get('label_colour', 'black')
        verts = []
        for c in lay.cell:
            col = c.column
            poslist = [tuple([p for p in n.pos])
                                for n in col.node]
            verts.append(tuple(poslist))
            if labels == 'column':
                col_label = label_fmt % col.index
            elif labels == 'cell':
                col_label = label_fmt % c.index
            else: col_label = None
            if col_label:
                ax.text(col.centre[0], col.centre[1], col_label,
                        clip_on = True, horizontalalignment = 'center',
                        color = label_colour)

        linewidth = kwargs.get('linewidth', 0.2)
        linecolour = kwargs.get('linecolour', 'black')
        colourmap = kwargs.get('colourmap', None)
        polys = collections.PolyCollection(verts,
                                           linewidth = linewidth,
                                           facecolors = [],
                                           edgecolors = linecolour,
                                           cmap = colourmap)
        ax.add_collection(polys)

        if 'value' in kwargs:
            vals = kwargs['value']
            if len(vals) >= self.num_cells:
                vals = np.array(kwargs['value'])
                indices = [c.index for c in lay.cell]
                layer_vals = vals[indices]
                polys.set_array(layer_vals)
                self.plot_colourbar(ax, polys, kwargs)
                if labels == 'value':
                    for c in lay.cell:
                        col = c.column
                        col_label = label_fmt % vals[c.index]
                        ax.text(col.centre[0], col.centre[1], col_label,
                                clip_on = True, horizontalalignment = 'center',
                                color = label_colour)
            else:
                raise Exception('Not enough values for mesh in layer_plot()')

        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))

        ax.set_aspect(kwargs.get('aspect', 'equal'))
        ax.autoscale_view()

        if 'axes' not in kwargs: plt.show()

    def plot_colourbar(self, ax, polys, kwargs):
        """Adds colour bar to Matplotlib plot on specified axes."""

        import matplotlib.pyplot as plt

        cbar = plt.colorbar(polys, ax = ax)
        if 'value_label' in kwargs:
            value_label = kwargs.get('value_label')
            if 'value_unit' in kwargs:
                unit = kwargs.get('value_unit')
                value_label += ' (' + unit + ')'
            cbar.set_label(value_label)

    def slice_plot(self, line = 'x', **kwargs):
        """Creates a 2-D Matplotlib plot of the mesh through a specified
        vertical slice."""

        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if 'axes' in kwargs: ax = kwargs['axes']
        else: fig, ax = plt.subplots()

        if line == 'x':
            bounds = self.bounds
            l = ([bounds[0][0], self.centre[1]],
                 [bounds[1][0], self.centre[1]])
        elif line == 'y':
            bounds = self.bounds
            l = ([self.centre[0], bounds[0][1]],
                 [self.centre[0], bounds[1][1]])
        elif isinstance(line, (float, int)):
            # line through mesh centre at specified angle in degrees:
            bounds = self.bounds
            r = 0.5 * np.linalg.norm(bounds[1] - bounds[0])
            from math import radians, cos, sin
            theta = radians(linespec)
            d = r * np.array([sin(theta), cos(theta)])
            l = [self.centre - d, self.centre + d]
        else: l = line

        l = [np.array(p) for p in l]

        if np.linalg.norm(l[1] - l[0]) > 0.0:

            labels = kwargs.get('label', None)
            label_fmt = kwargs.get('label_format', '%g')
            label_colour = kwargs.get('label_colour', 'black')
            slice_cells = []
            verts = []
            dcol = {}
            track = self.column_track(l)
            for item in track:
                col, points = item[0], item[1:]
                inpoint = points[0]
                if len(points) > 1: outpoint = points[1]
                else: outpoint = inpoint
                if line == 'x':
                    din, dout = inpoint[0], outpoint[0]
                    default_xlabel = 'x'
                elif line == 'y':
                    din, dout = inpoint[1], outpoint[1]
                    default_xlabel = 'y'
                else:
                    din = np.linalg.norm(inpoint - l[0])
                    dout = np.linalg.norm(outpoint - l[0])
                    default_xlabel = 'distance'
                dcol[col.index] = 0.5 * (din + dout)
                for c in col.cell:
                    slice_cells.append(c)
                    verts.append(((din, c.layer.bottom),
                                  (din, c.layer.top),
                                  (dout, c.layer.top),
                                  (dout, c.layer.bottom)))
                    if labels == 'cell':
                        cell_label = label_fmt % c.index
                    else: cell_label = None
                    if cell_label:
                        ax.text(dcol[col.index], c.layer.centre, cell_label,
                                clip_on = True, horizontalalignment = 'center',
                                color = label_colour)

            linewidth = kwargs.get('linewidth', 0.2)
            linecolour = kwargs.get('linecolour', 'black')
            colourmap = kwargs.get('colourmap', None)
            polys = collections.PolyCollection(verts,
                                               linewidth = linewidth,
                                               facecolors = [],
                                               edgecolors = linecolour,
                                               cmap = colourmap)
            ax.add_collection(polys)

            if 'value' in kwargs:
                vals = kwargs['value']
                if len(vals) >= self.num_cells:
                    vals = np.array(kwargs['value'])
                    indices = [c.index for c in slice_cells]
                    slice_vals = vals[indices]
                    polys.set_array(slice_vals)
                    self.plot_colourbar(ax, polys, kwargs)
                    if labels == 'value':
                        for c in slice_cells:
                            col = c.column
                            cell_label = label_fmt % vals[c.index]
                            ax.text(dcol[col.index], c.layer.centre, cell_label,
                                    clip_on = True, horizontalalignment = 'center',
                                    color = label_colour)
                else:
                    raise Exception('Not enough values for mesh in slice_plot()')

            ax.set_xlabel(kwargs.get('xlabel', default_xlabel))
            ax.set_ylabel(kwargs.get('ylabel', 'z'))

            ax.set_aspect(kwargs.get('aspect', 'auto'))
            ax.autoscale_view()

            if 'axes' not in kwargs: plt.show()

        else:
            raise Exception('Line of zero length in slice_plot()')

    def fit_data_to_columns(self, data, columns = None, smoothing = 0.01):
        """Fits scattered data to the columns of the mesh, using smoothed
        piecewise constant least-squares fitting.

        The data should be in the form of a 3-column array with x,y,z
        data in each row.

        Fitting can be carried out over a subset of the mesh columns
        by specifying a list or array of column indices.

        Increasing the smoothing parameter will decrease gradients
        between columns, and a non-zero value must be used to obtain a
        solution if any columns contain no data.

        """

        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        if columns is None: columns = self.column
        col_dict = {col.index: i for i, col in enumerate(columns)}

        N = len(columns)
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)

        for d in data:
            col = self.find(d[:2])
            if col:
                if col.index in col_dict:
                    i = col_dict[col.index]
                    A[i,i] += 1
                    b[i] += d[2]

        faces = self.column_faces(columns)
        for f in faces:
            cols = tuple(f)
            i = [col_dict[col] for col in cols]
            A[i[0], i[0]] += smoothing
            A[i[0], i[1]] -= smoothing
            A[i[1], i[1]] += smoothing
            A[i[1], i[0]] -= smoothing

        A = A.tocsr()
        return spsolve(A, b)

    def fit_surface(self, data, columns = None, smoothing = 0.01):
        """Fits surface elevation data to determine the number of layers in
        each column.

        The data should be in the form of a 3-column array with x,y,z
        data in each row.

        Fitting can be carried out over a subset of the mesh columns
        by specifying a list or array of columns.

        Increasing the smoothing parameter will decrease gradients
        between columns, and a non-zero value must be used to obtain a
        solution if any columns contain no data."""

        if columns is None: columns = self.column

        z = self.fit_data_to_columns(data, columns, smoothing)

        for col, s in zip(columns, z):
            col.set_surface(self.layer, s)
        self.setup()
