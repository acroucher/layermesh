import numpy as np
import unittest
from layermesh import mesh

class meshTestCase(unittest.TestCase):

    def test_rectangular(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh(columns = [dx, dy], layers = dz)
        self.assertEqual(m.num_nodes, 16)
        self.assertEqual(m.num_columns, 9)
        self.assertEqual(m.num_layers, 3)
        self.assertEqual(m.num_cells, 27)

        self.assertEqual(np.linalg.norm(
            m.node[0].pos), 0.)
        self.assertEqual(np.linalg.norm(
            m.node[-1].pos -
            np.array([30., 36.])), 0.)

        centroids = [[5, 6], [15, 6], [25, 6],
                     [5, 18], [15, 18], [25, 18],
                     [5, 30], [15, 30], [25, 30]]
        for col, centroid in zip(m.column, centroids):
            self.assertEqual(
                np.linalg.norm(col.centre - np.array(centroid)), 0.)
        layer_centres = [-0.5, -2, -4.5]
        for lay, centre in zip(m.layer, layer_centres):
            self.assertEqual(lay.centre, centre)
        area = 120.
        self.assertTrue(all([col.area == area for col in m.column]))
        vol = 120. * 6
        self.assertTrue(all([col.volume == vol for col in m.column]))
        self.assertEqual(m.area, 1080)
        self.assertEqual(m.volume, 6480)

    def test_surface(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh(columns = [dx, dy], layers = dz)
        self.assertEqual([c.index for c in m.surface_cells], range(9))

        surface = {4: -1}
        m = mesh(columns = [dx, dy], layers = dz, surface = surface)
        self.assertEqual(m.num_cells, 26)
        self.assertEqual([col.num_cells for col in m.column],
                         [3, 3, 3, 3, 2, 3, 3, 3, 3])
        self.assertEqual([lay.num_cells for lay in m.layer],
                         [8, 9, 9])
        self.assertEqual(m.volume, 6360)
        self.assertEqual([c.index for c in m.surface_cells],
                         [0, 1, 2, 3, 12, 4, 5, 6, 7])

        surface = [-3] * 9
        m = mesh(columns = [dx, dy], layers = dz, surface = surface)
        self.assertEqual(m.num_cells, 9)
        self.assertEqual([col.num_cells for col in m.column], [1] * 9)
        self.assertEqual([lay.num_cells for lay in m.layer], [0, 0, 9])
        self.assertEqual(m.volume, 3240)
        self.assertEqual([c.index for c in m.surface_cells], range(9))

        surface = [0.2, -0.8, -1.5] * 3
        m = mesh(columns = [dx, dy], layers = dz, surface = surface)
        self.assertEqual([col.num_cells for col in m.column], [3, 2, 2] * 3)
        self.assertEqual([lay.num_cells for lay in m.layer], [3, 9, 9])
        self.assertEqual(m.volume, 5760)
        self.assertEqual([c.index for c in m.surface_cells],
                         [0, 4, 5, 1, 7, 8, 2, 10, 11])

    def test_meshio_points_cells(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh(columns = [dx, dy], layers = dz)

        points, cells = m.meshio_points_cells
        self.assertEqual(len(points), 16 * 4)
        self.assertEqual(len(cells['hexahedron']), 9 * 3)

        surface = [0.2, -0.8, -1.5] * 3
        m = mesh(columns = [dx, dy], layers = dz, surface = surface)
        points, cells = m.meshio_points_cells
        self.assertEqual(len(cells['hexahedron']), 21)

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(meshTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)


