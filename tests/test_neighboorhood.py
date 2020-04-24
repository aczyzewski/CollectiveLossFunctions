import unittest
import torch

from src.neighboorhood import FaissKNN


class TestNeighboorhood(unittest.TestCase):

    def test_knn_wrapper(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
