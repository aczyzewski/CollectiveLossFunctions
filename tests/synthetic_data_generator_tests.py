import unittest
from src.data.generator import SyntheticDataGenerator

class TestSyntheticDataGenerator(unittest.TestCase):

    def test_custom_gen_function(self):
        """ Tests if provided function is used during data generation process """
        
        data = SyntheticDataGenerator(lambda x: 1)
        x, y = data.generate(100, noise_multiplier = 0)
        self.assertEqual(sum(y), 100.0)

    def test_range(self):
        """ Tests if range of returned points is correct """

        data = SyntheticDataGenerator()
        range_min, range_max = -1, 12
        x, y = data.generate(num_points = 1000, r_min = range_min, r_max = range_max)
        self.assertTrue(min(x) >= range_min and max(x) <= range_max)

    def test_size(self):
        """ Tests if number of returned points is correct """

        data = SyntheticDataGenerator()
        num_points = 15
        x, y = data.generate(num_points)
        self.assertEqual(len(x), len(y), num_points)

if __name__ == '__main__':
    unittest.main()