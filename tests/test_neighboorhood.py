import unittest
import numpy as np
from src.neighboorhood import FaissKNN


class TestNeighboorhood(unittest.TestCase):

    example_dataset = np.array([
        [0., 1., 0., 1., 1.],   # K = 2 (D = 2)
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],
        [1., 0., 1., 1., 1.],   # K = 3 (D = 3)
        [1., 0., 0., 0., 0.]    # K = 1 (D = 1)
    ])
    x, y = example_dataset[:, :-1], example_dataset[:, -1]

    def test_knn_output(self):
        knn = FaissKNN(self.x, self.y, k=3)
        query = np.array([[0., 0., 0., 0.]]).astype('float32')
        distances, indices, classes = knn.get(query, exclude_query=True)

        self.assertTrue((distances == np.array([[1., 2., 3.]])).all())
        self.assertTrue((indices == np.array([[4., 0., 3.]])).all())
        self.assertTrue((classes == np.array([[0., 1., 1.]])).all())

    def test_knn_precompute_output(self):
        knn = FaissKNN(self.x, self.y, k=3, precompute=True)
        query = np.array([2])
        distances, indices, classes = knn.get(query, exclude_query=True)

        self.assertTrue((distances == np.array([[1., 2., 3.]])).all())
        self.assertTrue((indices == np.array([[4., 0., 3.]])).all())
        self.assertTrue((classes == np.array([[0., 1., 1.]])).all())


if __name__ == '__main__':
    unittest.main()
