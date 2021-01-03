import unittest
#from src import algorithms
from src.algorithms import *
import numpy

class DUCRNAlgorithmsTest(unittest.TestCase):
    def test_reaction_number(self):
        self.assertEqual(0, ReactionNumber(numpy.zeros((3,3,3))))
        self.assertEqual(24, ReactionNumber(numpy.ones((3,3,3))))

    def test_bitlist_to_dec(self):
        self.assertEqual(0, Dec([0]*5))
        self.assertEqual(1, Dec([1]))
        self.assertEqual(1, Dec([0, 1]))
        self.assertEqual(657, Dec([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]))

    def test_binarize_noncore_edges(self):
        self.assertEqual([], Binarize(numpy.ones((3,3,3)), numpy.zeros((0,3))))
        self.assertEqual([1]*24, Binarize(numpy.ones((3,3,3)), numpy.array([(i,j,k) for i in range(3) for j in range(3) for k in range(3) if i != 0 or j != k])))
        self.assertEqual([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], Binarize(numpy.stack([numpy.eye(3)]*3), numpy.array([(i,j,k) for i in range(3) for j in range(3) for k in range(3) if i != 0 or j != k])))

    def test_find_complex(self):
        self.assertEqual(1, FindComplex(numpy.array([[]]), []))
        self.assertEqual(0, FindComplex(numpy.eye(2), numpy.array([[1,0]])))
        self.assertEqual(1, FindComplex(numpy.eye(2), numpy.array([[0,1]])))
        self.assertEqual(2, FindComplex(numpy.eye(2), numpy.array([[1,1]])))

    def test_find_row(self):
        self.assertEqual(1, FindComplex(numpy.array([[]]), []))
        self.assertEqual(0, FindRow(numpy.array([(i,j,k) for i in range(3) for j in range(3) for k in range(3) if i != 0 or j != k]), numpy.array([0,0,1])))
        self.assertEqual(23, FindRow(numpy.array([(i,j,k) for i in range(3) for j in range(3) for k in range(3) if i != 0 or j != k]), numpy.array([2,2,2])))
        self.assertEqual(24, FindRow(numpy.array([(i,j,k) for i in range(3) for j in range(3) for k in range(3) if i != 0 or j != k]), numpy.array([0,0,0])))
