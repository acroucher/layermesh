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
        self.assertTrue((np.array([-1.5, -1.1]) == bds[0]).all())
        self.assertTrue((np.array([4., 3.]) == bds[1]).all())

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
        tol = 1.e-6
        poly = [
            np.array([1., 2.]),
            np.array([3., 2.]),
            np.array([3., 4.]),
            np.array([1., 4.]),
        ]
        c = geometry.polygon_centroid(poly)
        self.assertTrue((np.abs(np.array([2., 3.]) - c) <= tol).all())
        poly = [
            np.array([0., 0.]),
            np.array([15., 0.]),
            np.array([15., 5.]),
            np.array([10., 5.]),
            np.array([10., 10.]),
            np.array([0., 10.])
        ]
        c = geometry.polygon_centroid(poly)
        self.assertTrue((np.abs(np.array([6.5, 4.5]) - c) <= tol).all())

    def test_line_polygon_intersect(self):
        """line_polygon_intersections()"""
        tol = 1.e-6
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
        self.assertTrue((np.abs(np.array([0., 0.]) - pts[0]) <= tol).all())
        self.assertTrue((np.abs(np.array([15., 4.]) - pts[1]) <= tol).all())
        line = [
            np.array([0., 17.5]),
            np.array([17.5, 0.])
        ]
        pts = geometry.line_polygon_intersections(poly, line)
        self.assertEqual(4, len(pts))
        self.assertTrue((np.abs(np.array([7.5, 10.]) - pts[0]) <= tol).all())
        self.assertTrue((np.abs(np.array([10., 7.5]) - pts[1]) <= tol).all())
        self.assertTrue((np.abs(np.array([12.5, 5.]) - pts[2]) <= tol).all())
        self.assertTrue((np.abs(np.array([15., 2.5]) - pts[3]) <= tol).all())

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
        tol = 1.e-6
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
        self.assertTrue((np.abs(np.array([5., 0.]) - b) <= tol).all())
        p1 = np.array([5., 2.5])
        p2 = np.array([20., 2.5])
        b = geometry.polygon_boundary(p1, p2, poly)
        self.assertTrue((np.abs(np.array([15., 2.5]) - b) <= tol).all())
        b = geometry.polygon_boundary(p2, p1, poly)
        self.assertTrue((np.abs(np.array([15., 2.5]) - b) <= tol).all())

    def test_line_projection_distance(self):
        """line_projection() and point_line_distance()"""
        from math import sqrt
        tol = 1.e-6
        line = [
            np.array([0., 0.]),
            np.array([1., 1.])
        ]
        a = np.array([1., 0.])
        p, xi = geometry.line_projection(a, line, True)
        self.assertTrue((np.abs(np.array([0.5, 0.5]) - p) <= tol).all())
        self.assertAlmostEqual(0.5, xi)
        d = geometry.point_line_distance(a, line)
        self.assertAlmostEqual(sqrt(0.5), d)
        a = np.array([2., 1.])
        p, xi = geometry.line_projection(a, line, True)
        self.assertTrue((np.abs(np.array([1.5, 1.5]) - p) <= tol).all())
        self.assertAlmostEqual(1.5, xi)
        d = geometry.point_line_distance(a, line)
        self.assertAlmostEqual(sqrt(0.5), d)

    def test_vector_heading(self):
        """vector_heading()"""
        from math import atan
        p = np.array([0., 10.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(0., h)

        p = np.array([1., 1.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(0.25 * np.pi, h)

        p = np.array([-2., -2.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(5. * np.pi / 4., h)

        p = np.array([-3., 4.])
        h = geometry.vector_heading(p)
        self.assertAlmostEqual(1.5 * np.pi + atan(4./ 3.), h)

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(geometryTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)
