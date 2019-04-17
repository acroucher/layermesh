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

    def test_meshio_points_cells(self):

        dx = [10.]*3; dy = [12.] * 3
        dz = [1., 2., 3.]
        m = mesh(columns = [dx, dy], layers = dz)

        points, cells = m.meshio_points_cells
        self.assertEqual(len(points), 16 * 4)
        self.assertEqual(len(cells['hexahedron']), 9 * 3)
        self.assertEqual(np.linalg.norm(points[0]), 0.)
        self.assertEqual(
            np.linalg.norm(points[-1] -
                           np.array([30., 36., -6])), 0.)
        self.assertEqual(list(cells['hexahedron'][0]),
                         [16, 20, 21, 17, 0, 4, 5, 1])
        self.assertEqual(list(cells['hexahedron'][-1]),
                         [58, 62, 63, 59, 42, 46, 47, 43])


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(meshTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)


