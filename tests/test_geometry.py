import numpy as np
import unittest
from layermesh import geometry

class geometryTestCase(unittest.TestCase):

    def test_in_polygon(self):
        """in_polygon()"""
        poly = [
            np.array([0., 0.]),
            np.array([1., 0.]),
            np.array([0., 1.]),
        ]
        p = np.array([0.8, 0.9])
        self.assertFalse(geometry.in_polygon(p, poly))
        p = np.array([0.4, 0.3])
        self.assertTrue(geometry.in_polygon(p, poly))

        poly = [[0, 0], [0, 1], [1, 1]]
        p = [0.5, 0.8]
        self.assertTrue(geometry.in_polygon(p, poly))

    def test_in_rectangle(self):
        """in_rectangle()"""
        rect = [
            np.array([5., 4.]),
            np.array([8., 11.]),
        ]
        p = np.array([0.8, 0.9])
        self.assertFalse(geometry.in_rectangle(p, rect))
        p = np.array([7., 10.5])
        self.assertTrue(geometry.in_rectangle(p, rect))

    def test_rect_intersect(self):
        """rectangles_intersect()"""
        r1 = [
            np.array([5., 4.]),
            np.array([8., 11.]),
        ]
        r2 = [
            np.array([7., 5.]),
            np.array([13., 10.]),
        ]
        self.assertTrue(geometry.rectangles_intersect(r1, r2))
        self.assertTrue(geometry.rectangles_intersect(r2, r1))
        r2 = [
            np.array([1., 2.]),
            np.array([4.6, 10.]),
        ]
        self.assertFalse(geometry.rectangles_intersect(r1, r2))
        self.assertFalse(geometry.rectangles_intersect(r2, r1))

    def test_bounds_of_points(self):
        """bounds_of_points()"""
        pts = [
            np.array([-1.5, 2.1]),
            np.array([1., -1.1]),
            np.array([4., 3.])
        ]
        bds = geometry.bounds_of_points(pts)
        self.assertTrue(np.allclose(bds[0], np.array([-1.5, -1.1])))
        self.assertTrue(np.allclose(bds[1], np.array([4., 3.])))

    def test_polygon_area(self):
        """polygon_area()"""
        poly = [
            np.array([0., 0.]),
            np.array([1., 0.]),
            np.array([0., 1.]),
        ]
        self.assertEqual(geometry.polygon_area(poly), 0.5)

        poly = [[1, 2], [3, 2], [3, 4], [1, 4]]
        self.assertEqual(geometry.polygon_area(poly), 4)

        poly = [[1., 2], [1, 4.], [3., 4], [3., 2]]
        self.assertEqual(geometry.polygon_area(poly), 4)

        poly = [[0, 0], [0, 1], [1, 1]]
        self.assertEqual(geometry.polygon_area(poly), 0.5)

        poly = [[0., 0.], [ 0., 20.], [10., 20.], [10.,  0.]]
        self.assertEqual(geometry.polygon_area(poly), 200.)

    def test_polygon_centroid(self):
        """polygon_centroid()"""
        poly = [
            np.array([1., 2.]),
            np.array([3., 2.]),
            np.array([3., 4.]),
            np.array([1., 4.]),
        ]
        c = geometry.polygon_centroid(poly)
        self.assertTrue(np.allclose(c, np.array([2, 3])))
        poly = [
            np.array([0., 0.]),
            np.array([15., 0.]),
            np.array([15., 5.]),
            np.array([10., 5.]),
            np.array([10., 10.]),
            np.array([0., 10.])
        ]
        c = geometry.polygon_centroid(poly)
        self.assertTrue(np.allclose(c, np.array([6.5, 4.5])))

    def test_line_polygon_intersect(self):
        """line_polygon_intersections()"""
        poly = [
            np.array([0., 0.]),
            np.array([15., 0.]),
            np.array([15., 5.]),
            np.array([10., 5.]),
            np.array([10., 10.]),
            np.array([0., 10.])
        ]
        line = [
            np.array([0., 0.]),
            np.array([15., 4.])
        ]
        pts = geometry.line_polygon_intersections(poly, line)
        self.assertEqual(2, len(pts))
        self.assertTrue(np.allclose(pts[0], np.zeros(2)))
        self.assertTrue(np.allclose(pts[1], np.array([15, 4])))
        line = [
            np.array([0., 17.5]),
            np.array([17.5, 0.])
        ]
        pts = geometry.line_polygon_intersections(poly, line)
        self.assertEqual(4, len(pts))
        self.assertTrue(np.allclose(pts[0], np.array([7.5, 10])))
        self.assertTrue(np.allclose(pts[1], np.array([10, 7.5])))
        self.assertTrue(np.allclose(pts[2], np.array([12.5, 5])))
        self.assertTrue(np.allclose(pts[3], np.array([15, 2.5])))

    def test_simplify_polygon(self):
        """simplify_polygon()"""
        poly = [
            np.array([0., 0.]),
            np.array([1., 0.]),
            np.array([1., 1.]),
            np.array([0.5, 1.]),
            np.array([0., 1.])
        ]
        s = geometry.simplify_polygon(poly)
        self.assertEqual(4, len(s))
        poly = [
            np.array([0., 0.]),
            np.array([10., 0.]),
            np.array([15., 0.]),
            np.array([15., 5.]),
            np.array([11., 5.]),
            np.array([10., 5.]),
            np.array([10., 10.]),
            np.array([2., 10.]),
            np.array([0., 10.])
        ]
        s = geometry.simplify_polygon(poly)
        self.assertEqual(6, len(s))

    def test_polygon_boundary(self):
        """polygon_boundary()"""
        poly = [
            np.array([0., 0.]),
            np.array([15., 0.]),
            np.array([15., 5.]),
            np.array([10., 5.]),
            np.array([10., 10.]),
            np.array([0., 10.])
        ]
        p1 = np.array([5., -1.])
        p2 = np.array([5., 12.])
        b = geometry.polygon_boundary(p1, p2, poly)
        self.assertTrue(np.allclose(b, np.array([5, 0])))
        p1 = np.array([5., 2.5])
        p2 = np.array([20., 2.5])
        b = geometry.polygon_boundary(p1, p2, poly)
        self.assertTrue(np.allclose(b, np.array([15., 2.5])))
        b = geometry.polygon_boundary(p2, p1, poly)
        self.assertTrue(np.allclose(b, np.array([15., 2.5])))

    def test_line_projection_distance(self):
        """line_projection() and point_line_distance()"""
        from math import sqrt
        line = [
            np.array([0., 0.]),
            np.array([1., 1.])
        ]
        a = np.array([1., 0.])
        p, xi = geometry.line_projection(a, line, True)
        self.assertTrue(np.allclose(p, np.array([0.5, 0.5])))
        self.assertEqual(xi, 0.5)
        d = geometry.point_line_distance(a, line)
        self.assertEqual(d, sqrt(0.5))
        a = np.array([2., 1.])
        p, xi = geometry.line_projection(a, line, True)
        self.assertTrue(np.allclose(p, np.array([1.5, 1.5])))
        self.assertEqual(xi, 1.5)
        d = geometry.point_line_distance(a, line)
        self.assertEqual(d, sqrt(0.5))

    def test_vector_heading(self):
        """vector_heading()"""
        from math import atan
        p = np.array([0., 10.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(h, 0.)

        p = np.array([1., 1.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(h, 0.25 * np.pi)

        p = np.array([-2., -2.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(h, 5. * np.pi / 4.)

        p = np.array([-3., 4.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(h, 1.5 * np.pi + atan(4./ 3.))

    def test_rotation(self):
        """rotation()"""
        from math import sqrt

        A, b = geometry.rotation(0.)
        self.assertTrue(np.allclose(A, np.identity(2)))
        self.assertTrue(np.allclose(b, np.zeros(2)))

        A, b = geometry.rotation(90.)
        self.assertTrue(np.allclose(A, np.array([[0, 1], [-1, 0]])))
        self.assertTrue(np.allclose(b, np.zeros(2)))

        A, b = geometry.rotation(-30., (0, 0))
        self.assertTrue(np.allclose(A, np.array([[sqrt(0.75), -0.5],
                                                 [0.5, sqrt(0.75)]])))
        self.assertTrue(np.allclose(b, np.zeros(2)))

        A, b = geometry.rotation(0., [1, 0])
        self.assertTrue(np.allclose(A, np.identity(2)))
        self.assertTrue(np.allclose(b, np.zeros(2)))

        A, b = geometry.rotation(90., [1, 1])
        self.assertTrue(np.allclose(A, np.array([[0, 1], [-1, 0]])))
        self.assertTrue(np.allclose(b, np.array([0, 2])))
        p = np.array([1, 1])
        r = np.dot(A, p) + b
        self.assertTrue(np.allclose(r, np.array([1, 1])))
        p = np.array([0, 0])
        r = np.dot(A, p) + b
        self.assertTrue(np.allclose(r, np.array([0, 2])))
        p = np.array([2, 1])
        r = np.dot(A, p) + b
        self.assertTrue(np.allclose(r, np.array([1, 0])))

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(geometryTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)
