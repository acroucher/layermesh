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
                 for col in m.column[index].side_neighbours]
        self.assertEqual(col_sidenbrs(0), [None, 3, 1, None])
        self.assertEqual(col_sidenbrs(5), [4, 8, None, 2])
        self.assertEqual(col_sidenbrs(4), [3, 7, 5, 1])

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
        self.assertEqual(len(cells['hexahedron']), 21)

    def test_find(self):

        dx = [10, 20, 30]; dy = [20, 15, 10]
        dz = [5, 10, 15]
        surface = [0.2, -9, -18] * 3
        m = mesh.mesh(rectangular = (dx, dy, dz), surface = surface)

        self.assertEqual(m.num_cells, 18)
        self.assertEqual(m.volume, 56250)

        p = (8, 5, -2)
        c = m.find(p)
        self.assertEqual(c.index, 0)

        p = [25, 25, -12]
        c = m.find(p, indices = True)
        self.assertEqual(c, 6)

        p = [25, 25, -4]
        c = m.find(p)
        self.assertIsNone(c)

        p = np.array([40, 50, -20])
        c = m.find(p)
        self.assertIsNone(c)

        c = m.find([5, 8], indices = True)
        self.assertEqual(0, c)

        c = m.find([-10, -10])
        self.assertIsNone(c)

        l = m.find(-25, indices = True)
        self.assertEqual(2, l)

        poly = [[8, -5], [11, 40], [20, 35], [40, 10], [20, -5]]
        cells = m.layer[-1].find(poly, indices = True)
        self.assertEqual(cells, [10, 13])
        cells = m.layer[1].find(poly, indices = True)
        self.assertEqual(cells, [4, 6])
        cells = m.layer[0].find(poly)
        self.assertEqual(cells, [])

        c = m.layer[-1].find([45., 42.], indices = True)
        self.assertEqual(c, 17)
        c = m.layer[1].find([45., 42.])
        self.assertIsNone(c)

        cells = m.layer[-1].find(lambda c: c.column.centre[1] > 30,
                                 indices = True)
        self.assertEqual(cells, [15, 16, 17])

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

        rect = [(0,0), (30, 35)]
        cells = m.cells_inside(rect)
        self.assertEqual(len(cells), 10)
        cells = m.cells_inside(rect, elevations = [-30, -20])
        self.assertEqual(len(cells), 4)

        m.translate((100, 0, 10))
        c = m.find([105, 8], indices = True)
        self.assertEqual(0, c)

        l = m.find(-15, indices = True)
        self.assertEqual(2, l)

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

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(meshTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)


