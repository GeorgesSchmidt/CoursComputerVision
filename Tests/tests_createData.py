import unittest
import cv2
import os
import numpy as np

from Faces.createData import CreateDatas

class TestCreateDatas(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by specifying the directory paths and initializing the CreateDatas class.
        This assumes there are 'People' and 'King' subdirectories under './TestPictures' containing test images.
        """
        self.directory = './Pictures'
        
        # Creating directories and adding a few test images
        os.makedirs(os.path.join(self.directory, 'People'), exist_ok=True)
        os.makedirs(os.path.join(self.directory, 'King'), exist_ok=True)
        
        self.create_datas = CreateDatas(self.directory)
        
    def test_initialization(self):
        """
        Test that the initialization properly reads images, generates embeddings, and ensures the correct label structure.
        Specifically:
        - len(self.X) == len(self.y)
        - len(self.X[0]) == 512 (for embedding size)
        - self.y contains only 0 or 1 integers.
        """
        # Check that the lengths of X and y are the same
        self.assertEqual(len(self.create_datas.X), len(self.create_datas.y), "X and y should have the same length")

        # Check that the first embedding in X has 158 features (i.e., the expected size of the embeddings)
        self.assertEqual(len(self.create_datas.X[0]), 512, "Each embedding should have 512 dimensions")

        # Check that y contains only 0s and 1s
        for label in self.create_datas.y:
            self.assertIn(label, [0, 1], "y should only contain 0 or 1 integers")
        
    def test_get_datas(self):
        """
        Test that the get_datas method correctly retrieves embeddings from the given directory.
        """
        people_embeddings = self.create_datas.get_datas(os.path.join(self.directory, 'People'))
        king_embeddings = self.create_datas.get_datas(os.path.join(self.directory, 'King'))
        
        # Ensure that at least one embedding is retrieved from each category
        self.assertGreaterEqual(len(people_embeddings), 1)
        self.assertGreaterEqual(len(king_embeddings), 1)
        

if __name__ == '__main__':
    unittest.main()