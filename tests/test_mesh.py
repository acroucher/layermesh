import numpy as np
import unittest
from layermesh import mesh

class meshTestCase(unittest.TestCase):

    def test_null(self):

        m = mesh.mesh()
        self.assertEqual(m.node, [])
        self.assertEqual(m.column, [])
        self.assertEqual(m.layer, [])
        self.assertEqual(m.cell, [])
        self.assertEqual(m.volume, 0)

    def test_rectangular(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh.mesh(rectangular = (dx, dy, dz))
        self.assertEqual(m.num_nodes, 16)
        self.assertEqual(m.num_columns, 9)
        self.assertEqual(m.num_layers, 3)
        self.assertEqual(m.num_cells, 27)
        faces = m.column_faces()
        self.assertEqual(len(faces), 12)
        self.assertEqual(len(m.type_columns(4)), 9)
        self.assertEqual(m.type_columns(3), [])

        self.assertTrue(np.allclose(m.node[0].pos, np.zeros(2)))
        self.assertTrue(np.allclose(m.node[-1].pos, np.array([30., 36.])))

        centroids = [[5, 6], [15, 6], [25, 6],
                     [5, 18], [15, 18], [25, 18],
                     [5, 30], [15, 30], [25, 30]]
        for col, centroid in zip(m.column, centroids):
            self.assertTrue(np.allclose(col.centre, np.array(centroid)))
        layer_centres = [-0.5, -2, -4.5]
        for lay, centre in zip(m.layer, layer_centres):
            self.assertEqual(lay.centre, centre)
        area = 120.
        self.assertTrue(all([col.area == area for col in m.column]))
        vol = 120. * 6
        self.assertTrue(all([col.volume == vol for col in m.column]))
        self.assertEqual(m.area, 1080)
        self.assertEqual(m.volume, 6480)
        self.assertEqual(np.linalg.norm(m.centre - np.array([15., 18.])), 0.)

        self.assertEqual([col.num_neighbours for col in m.column],
                          [2, 3, 2, 3, 4, 3, 2, 3, 2])
        def colnbrs(index):
            n = [col.index for col in m.column[index].neighbour]
            n.sort()
            return n
        self.assertEqual(colnbrs(0), [1, 3])
        self.assertEqual(colnbrs(4), [1, 3, 5, 7])
        self.assertEqual(colnbrs(7), [4, 6, 8])

        def col_sidenbrs(index):
            return [None if col is None else col.index
                 for col in m.column[index].side_neighbour]
        self.assertEqual(col_sidenbrs(0), [None, 3, 1, None])
        self.assertEqual(col_sidenbrs(5), [4, 8, None, 2])
        self.assertEqual(col_sidenbrs(4), [3, 7, 5, 1])

        self.assertEqual(12, len(m.boundary_nodes))
        m.delete_column(m.column[4])
        m.setup(indices = True)
        self.assertEqual(16, len(m.boundary_nodes))

        m = mesh.mesh(rectangular = (dx, dy, dz))
        for col in m.column:
            a = col.interior_angle
            self.assertTrue(np.allclose(a, 0.5 * np.pi))
            self.assertEqual(1, col.angle_ratio)
            self.assertEqual(dy[0] / dx[0], col.face_length_ratio)

        self.assertIsNone(m.layer[0].above)
        self.assertEqual(m.layer[0].below, m.layer[1])
        self.assertEqual(m.layer[1].above, m.layer[0])
        self.assertEqual(m.layer[1].above.below, m.layer[1])
        self.assertEqual(m.layer[2].above, m.layer[1])
        self.assertIsNone(m.layer[2].below)

        self.assertEqual(m.cell[0].num_neighbours, 3)
        self.assertEqual(m.cell[1].num_neighbours, 4)
        self.assertEqual(m.cell[-1].num_neighbours, 3)

        self.assertIsNone(m.cell[0].above)
        self.assertEqual(m.cell[0].below.index, m.cell[9].index)
        self.assertEqual(m.cell[9].above.index, m.cell[0].index)
        self.assertIsNone(m.cell[-1].below)

    def test_surface(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh.mesh(rectangular = (dx, dy, dz))
        self.assertEqual([c.index for c in m.surface_cells], list(range(9)))

        surface = {4: -1}
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        self.assertEqual(m.num_cells, 26)
        self.assertEqual([col.num_cells for col in m.column],
                         [3, 3, 3, 3, 2, 3, 3, 3, 3])
        self.assertEqual([lay.num_cells for lay in m.layer],
                         [8, 9, 9])
        self.assertEqual(m.volume, 6360)
        self.assertEqual([c.index for c in m.surface_cells],
                         [0, 1, 2, 3, 12, 4, 5, 6, 7])

        surface = [-3] * 9
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        self.assertEqual(m.num_cells, 9)
        self.assertEqual([col.num_cells for col in m.column], [1] * 9)
        self.assertEqual([lay.num_cells for lay in m.layer], [0, 0, 9])
        self.assertEqual(m.volume, 3240)
        self.assertEqual([c.index for c in m.surface_cells], list(range(9)))

        surface = [0.2, -0.8, -1.5] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        self.assertEqual([col.num_cells for col in m.column], [3, 2, 2] * 3)
        self.assertEqual([lay.num_cells for lay in m.layer], [3, 9, 9])
        self.assertEqual(m.volume, 5760)
        self.assertEqual([c.index for c in m.surface_cells],
                         [0, 4, 5, 1, 7, 8, 2, 10, 11])
        self.assertEqual([lay.area for lay in m.layer],
                         [360, 1080, 1080])
        self.assertEqual([lay.volume for lay in m.layer],
                         [360, 2160, 3240])
        self.assertTrue(all([c.surface for c in m.surface_cells]))
        subsurface_cells = list(set(m.cell) - set(m.surface_cells))
        self.assertFalse(any([c.surface for c in subsurface_cells]))

        self.assertEqual(m.cell[0].num_neighbours, 2)
        self.assertEqual(m.cell[1].num_neighbours, 3)
        self.assertEqual(m.cell[3].num_neighbours, 4)
        self.assertEqual(m.cell[7].num_neighbours, 5)
        self.assertEqual(m.cell[-1].num_neighbours, 3)

    def test_translate(self):

        dx = [10, 20, 30]; dy = [20, 15, 10]
        dz = [5, 10, 15]
        surface = [0.2, -9, -18] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        b = m.bounds
        self.assertTrue(np.allclose((b[0]), np.zeros(2)))
        self.assertTrue(np.allclose(b[1], np.array([60, 45])))
        m.translate([10, 15, 30])
        b = m.bounds
        self.assertTrue(np.allclose(b[0], np.array([10, 15])))
        self.assertTrue(np.allclose(b[1], np.array([70, 60])))
        self.assertEqual(m.layer[0].top, 30)
        self.assertEqual(m.layer[-1].bottom, 0)
        self.assertEqual(m.layer[0].centre, 27.5)
        self.assertEqual(m.layer[-1].centre, 7.5)
        self.assertTrue(np.allclose(m.centre, np.array([40., 37.5])))

    def test_rotate(self):

        dx = [10, 20, 30]; dy = [20, 15, 10]
        dz = [5, 10, 15]
        surface = [0.2, -9, -18] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        m.rotate(90, np.zeros(2))
        b = m.bounds
        self.assertTrue(np.allclose(b[0], np.array([0, -60])))
        self.assertTrue(np.allclose(b[1], np.array([45, 0])))
        self.assertTrue(np.allclose(m.centre, np.array([22.5, -30])))
        self.assertTrue(np.allclose(m.column[-1].centre, np.array([40, -45])))

        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        m.rotate(180)
        b = m.bounds
        self.assertTrue(np.allclose(b[0], np.zeros(2)))
        self.assertTrue(np.allclose(b[1], np.array([60, 45])))
        self.assertTrue(np.allclose(m.centre, np.array([30, 22.5])))
        self.assertTrue(np.allclose(m.column[-1].centre, np.array([15, 5])))

    def test_meshio_points_cells(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh.mesh(rectangular = (dx, dy,dz))

        points, cells = m.meshio_points_cells
        self.assertEqual(len(points), 16 * 4)
        self.assertEqual(len(cells['hexahedron']), 9 * 3)

        surface = [0.2, -0.8, -1.5] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)
        points, cells = m.meshio_points_cells
        self.assertEqual(len(points), 14 * 4)
        self.assertEqual(len(cells['hexahedron']), 21)

    def test_find(self):

        poly = [[8, -5], [11, 40], [20, 35], [40, 10], [20, -5]]

        # null mesh:
        m = mesh.mesh()
        self.assertEqual(m.find(lambda c: True), [])
        self.assertIsNone(m.find(-1))
        self.assertIsNone(m.find((0,0)))
        self.assertEqual(m.find(poly), [])
        self.assertIsNone(m.find((0,0,0)))

        # rectangular mesh:
        dx = [10, 20, 30]; dy = [20, 15, 10]
        dz = [5, 10, 15]
        surface = [0.2, -9, -18] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)

        self.assertEqual(m.num_cells, 18)
        self.assertEqual(m.volume, 56250)

        # mesh match function:
        cells = m.find(lambda c: c.index == 0)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].index, 0)

        cells = m.find(lambda c: c.index >= 0, indices = True)
        self.assertEqual(cells, list(range(18)))

        cells = m.find(lambda c: c.index < 0)
        self.assertEqual(cells, [])

        cells = m.find(lambda c: c.centre[2] < -15, indices = True)
        self.assertEqual(cells, list(range(9, 18)))

        cells = m.find(lambda c: c.volume >= 4000, indices = True)
        self.assertEqual(cells, [4, 10, 11, 13, 14, 17])

        cells = m.find(lambda c: not c.surface)
        self.assertEqual(len(cells), 9)

        # mesh elevation:
        lay = m.find(-25, indices = True)
        self.assertEqual(2, lay)
        lay = m.find(5)
        self.assertIsNone(lay)

        # mesh 2-D point:
        c = m.find([5, 8], indices = True)
        self.assertEqual(0, c)

        c = m.find([-10, -10])
        self.assertIsNone(c)

        p = (8, 5, -2)
        c = m.find(p)
        self.assertEqual(c.index, 0)

        # mesh 3-D point:
        c = m.find([25, 25, -12], indices = True)
        self.assertEqual(c, 6)

        c = m.find([25, 25, -4])
        self.assertIsNone(c)

        c = m.find(np.array([40, 50, -20]))
        self.assertIsNone(c)

        # mesh polygon:
        cols = m.find(poly, indices = True, sort = True)
        self.assertEqual(cols, [1, 4])
        cols = m.layer[1].find(poly, indices = True, sort = True)
        self.assertEqual(cols, [1, 4])
        cells = m.layer[0].find(poly)
        self.assertEqual(cells, [])

        # layer function:
        cells = m.layer[-1].find(lambda c: c.column.centre[1] > 30,
                                 indices = True)
        self.assertEqual(cells, [15, 16, 17])

        # layer elevation:
        self.assertIsNone(m.layer[1].find(-17))
        self.assertIsNotNone(m.layer[0].find(-4))
        self.assertIsNotNone(m.layer[0].find(-0.1))

        # layer 2-D point:
        col = m.layer[-1].find([45., 42.], indices = True)
        self.assertEqual(col, 8)
        col = m.layer[1].find([45., 42.])
        self.assertIsNone(col)

        # layer polygon:
        cols = m.layer[0].find(poly)
        self.assertEqual(cols, [])
        cols = m.layer[1].find(poly, indices = True, sort = True)
        self.assertEqual(cols, [1, 4])

        # layer 3-D point:
        p = [17, 40, -13]
        c = m.layer[1].find(p)
        self.assertIsNotNone(c)
        self.assertIsNotNone(c.find(p))

        p = np.array([42, 40, -13])
        c = m.layer[1].find(p)
        self.assertIsNone(c)

        # column function:
        col = m.column[3]
        cells = col.find(lambda c: c.centre[2] < -6, sort = True, indices = True)
        self.assertEqual(cells, [5, 12])

        cells = col.find(lambda c: c.volume < 1000 , sort = True, indices = True)
        self.assertEqual(cells, [1])

        # column elevation:
        col = m.column[7]
        lay = col.find(-20)
        self.assertEqual(lay.index, 2)
        lay = col.find(-7, indices = True)
        self.assertEqual(lay, 1)
        lay = col.find(-4)
        self.assertIsNone(lay)
        lay = col.find(-4, indices = True)
        self.assertIsNone(lay)

        # column 2-D point:
        self.assertIsNone(col.find((25, 20)))
        self.assertIsNotNone(col.find([25, 38]))

        # column polygon:
        self.assertIsNone(m.column[7].find(poly))
        self.assertIsNone(m.column[2].find(poly))
        self.assertIsNotNone(m.column[1].find(poly))
        self.assertIsNotNone(m.column[4].find(poly))

        # column 3-D point:
        c = col.find((11, 40, -10))
        self.assertIsNotNone(c)
        self.assertEqual(c.index, 8)
        c = col.find((11, 40, -4))
        self.assertIsNone(c)

        # cell function:
        c = m.cell[14]
        f = lambda c: c.volume > 4000
        self.assertIsNotNone(c.find(f))
        f = lambda c: c.volume < 4000
        self.assertIsNone(c.find(f))

        # cell elevation:
        self.assertIsNone(c.find(-9))
        self.assertIsNotNone(c.find(-19))

        # cell 2-D point:
        self.assertIsNone(c.find((30, 18)))
        self.assertIsNotNone(c.find((50, 31)))

        # cell polygon:
        self.assertIsNone(m.cell[14].find(poly))
        self.assertIsNotNone(m.cell[4].find(poly))

        # cell 3-D point:
        self.assertIsNone(c.find([30, 18, -20]))
        self.assertIsNotNone(c.find((50, 31, -29)))

        # node:
        self.assertIsNone(m.node[2].find(poly))
        self.assertIsNotNone(m.node[5].find(poly))

        # transform mesh to test quadtree:
        m.translate((100, 0, 10))
        c = m.find([105, 8], indices = True)
        self.assertEqual(0, c)
        c = m.find([132, 40], indices = True)
        self.assertEqual(8, c)

        col = m.layer[1].find((120, 30), indices = True)
        self.assertEqual(col, 4)
        col = m.layer[1].find((20, 30))
        self.assertIsNone(col)

        l = m.find(-15, indices = True)
        self.assertEqual(2, l)

        m.translate((-100, 0, -10))
        m.rotate(90, [0, 0])
        col = m.find([30, -50], indices = True)
        self.assertEqual(5, col)
        c = m.find([8, -8, -20])
        self.assertEqual(c.index, 9)
        c = m.find([8, -8, -10], indices = True)
        self.assertEqual(c, 3)

    def test_column_track(self):

        dx = [10] * 3; dy = [20] * 4
        dz = [10] * 4
        m = mesh.mesh(rectangular = (dx, dy, dz))

        def track_indices(t): return [item[0].index for item in t]

        t = m.column_track(([0, 10], [30, 10]))
        self.assertEqual(track_indices(t), [0, 1, 2])
        self.assertTrue(np.allclose(t[0][1], np.array([0, 10])))
        self.assertTrue(np.allclose(t[0][2], np.array([10, 10])))

        t = m.column_track(([-5, 35], [18, 35]))
        self.assertEqual(track_indices(t), [3, 4])
        self.assertTrue(np.allclose(t[0][1], np.array([0, 35])))
        self.assertTrue(np.allclose(t[0][2], np.array([10, 35])))
        self.assertTrue(np.allclose(t[-1][1], np.array([10, 35])))
        self.assertTrue(np.allclose(t[-1][2], np.array([18, 35])))

        t = m.column_track(([5, 0], [5, 80]))
        self.assertEqual(track_indices(t), [0, 3, 6, 9])
        self.assertTrue(np.allclose(t[-1][1], np.array([5, 60])))
        self.assertTrue(np.allclose(t[-1][2], np.array([5, 80])))

        t = m.column_track(([0, 0], [30, 80]))
        self.assertEqual(track_indices(t), [0, 3, 4, 7, 8, 11])
        self.assertTrue(np.allclose(t[0][1], np.array([0, 00])))
        self.assertTrue(np.allclose(t[-1][2], np.array([30, 80])))

        t = m.column_track(([29, 1], [3, 71.5]))
        self.assertEqual(track_indices(t), [2, 5, 4, 7, 6, 9])

        t = m.column_track(([-2, 3.5], [40, 15]))
        self.assertEqual(track_indices(t), [0, 1, 2])

    def test_io(self):

        import os
        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        s = [0, 0, -1.5, -1.8, -2.1, -2.8, -3, -1, 0]
        m1 = mesh.mesh(rectangular = (dx, dy, dz), surface = s)
        refine_cols = m1.find([(0, 0), (10, 10)])
        m1.refine(refine_cols)
        filename = 'mesh.h5'
        m1.write(filename)

        m2 = mesh.mesh(filename)
        self.assertEqual(m1.cell_type_sort, m2.cell_type_sort)
        def nodepos(m): return np.array([n.pos for n in m.node])
        self.assertTrue(np.allclose(nodepos(m1), nodepos(m2)))
        def cols(m):
            maxn = max([col.num_nodes for col in m.column])
            d = np.full((m.num_columns, maxn), -1, dtype = int)
            for i, col in enumerate(m.column):
                d[i, 0: col.num_nodes] = [n.index for n in col.node]
            return d
        self.assertTrue(np.allclose(cols(m1), cols(m2)))
        def lays(m): return np.array([m.layer[0].top] + \
                         [lay.bottom for lay in m.layer])
        self.assertTrue(np.allclose(lays(m1), lays(m2)))
        def cells(m): return np.array([c.index for c in m.cell])
        self.assertTrue(np.allclose(cells(m1), cells(m2)))

        os.remove(filename)

    def test_fit(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1.]*3 + [2.] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz))

        def f(pos): return -(0.1 * pos[0] + 0.01 * pos[0] * pos[1])
        colpos = np.array([col.centre for col in m.column])
        z = np.array([f(p) for p in colpos])
        data = np.hstack((colpos, np.vstack(z)))
        zfit = m.fit_data_to_columns(data)
        self.assertTrue(np.allclose(z, zfit, rtol = 0.05))

        sub = [0, 2, 3, 6, 7]
        subdata = data[sub, :]
        zfit = m.fit_data_to_columns(subdata)
        self.assertTrue(np.allclose(z[sub], zfit[sub], rtol = 0.05))

        zfit = m.fit_data_to_columns(subdata, smoothing = 0.01)
        self.assertTrue(np.allclose(z[sub], zfit[sub], rtol = 0.05))

        zfit = m.fit_data_to_columns(subdata, smoothing = 0.02)
        self.assertFalse(np.allclose(z[sub], zfit[sub], rtol = 0.05))
        self.assertTrue(np.allclose(z[sub], zfit[sub], rtol = 0.06))

        z = -1
        data = np.array([[col.centre[0], col.centre[1], z]
                         for col in m.column])
        m.fit_surface(data)
        self.assertTrue(np.allclose(z, m.surface))

        subdata = data[sub, :]
        m.fit_surface(subdata)
        self.assertTrue(np.allclose(z, m.surface))

        x = 20
        z2 = -2
        cols = [col for col in m.column if col.centre[0] < x]
        data = np.array([[col.centre[0], col.centre[1], z2]
                         for col in cols])
        m.fit_surface(data, columns = cols)
        expected = np.array([[col.centre[0], col.centre[1],
                              z2 if col.centre[0] < x else z]
                             for col in m.column])
        self.assertTrue(np.allclose(expected[:, 2], m.surface))

    def test_refine(self):

        def cell_types_in_order(m):
            cell_types = [c.num_nodes for c in m.cell]
            if m.cell_type_sort < 0:
                return all([x >= cell_types[i+1]
                            for i,x in enumerate(cell_types[:-1])])
            elif m.cell_type_sort > 0:
                return all([x <= cell_types[i+1]
                            for i,x in enumerate(cell_types[:-1])])
            else: return True

        dx, dy, dz = [100]*10, [150]*8, [10]*3

        m = mesh.mesh(rectangular = (dx, dy, dz))
        original_area = m.area
        original_centre = m.centre
        m.refine()
        self.assertEqual(m.area, original_area)
        self.assertTrue(np.allclose(m.centre, original_centre))
        self.assertEqual(m.num_columns, 20 * 16)
        self.assertEqual(m.num_nodes, 21 * 17)
        self.assertEqual(len(m.type_columns(3)), 0)
        self.assertTrue(cell_types_in_order(m))

        m = mesh.mesh(rectangular = (dx, dy, dz))
        cols = [col for col in m.column if col.centre[0] < 200]
        m.refine(cols)
        self.assertEqual(m.area, original_area)
        self.assertTrue(np.allclose(m.centre, original_centre))
        self.assertEqual(m.num_columns, 16 * 4 + 8 * 3 + 7 * 8)
        self.assertEqual(m.num_nodes, 5 * 17 + 8 * 9)
        self.assertEqual(len(m.type_columns(3)), 24)
        self.assertTrue(cell_types_in_order(m))

        m = mesh.mesh(rectangular = (dx, dy, dz))
        cols = m.find([(0,0), (400, 500)])
        self.assertEqual(len(cols), 12)
        m.refine(cols)
        self.assertEqual(m.area, original_area)
        self.assertTrue(np.allclose(m.centre, original_centre))
        self.assertEqual(m.num_columns, 12 * 4 + 7 * 3 + 16 + 5 + 8 * 5)
        self.assertEqual(m.num_nodes, 9 * 7 + 5 * 5 + 9 * 6)
        self.assertEqual(len(m.type_columns(3)), 21)
        self.assertTrue(cell_types_in_order(m))

        m = mesh.mesh(rectangular = (dx, dy, dz))
        m.rotate(30)
        cols = [col for col in m.column if 400. < col.centre[1] < 600.]
        m.refine(cols)
        self.assertEqual(m.area, original_area)
        self.assertEqual(m.num_columns, 189)
        self.assertEqual(m.num_nodes, 169)
        self.assertTrue(cell_types_in_order(m))

    def test_optimize(self):

        dx, dy, dz = [100]*10, [150]*8, [10]*3
        m = mesh.mesh(rectangular = (dx, dy, dz))
        original_area = m.area

        cols = m.find([(400, 400), (600, 600)])
        m.refine(cols)

        triangles = [col for col in m.column if col.num_nodes == 3]
        m.optimize(columns = triangles)

        self.assertTrue(np.allclose(m.area, original_area))
        faces = m.column_faces()
        cosines = np.array([f.angle_cosine for f in faces])
        self.assertTrue(np.max(np.abs(cosines)) < 0.1)

        p = (500, 500, -24)
        c = m.find(p)
        self.assertIsNotNone(c)
        self.assertIsNotNone(c.find(p))

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(meshTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)

