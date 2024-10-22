import unittest
import cv2
import os
import numpy as np
from Faces.detectFaces import DetectFaces

class TestDetectFaces(unittest.TestCase):

    def setUp(self):
        """
        Set up the test by creating an instance of DetectFaces and preparing the test images.
        """
        self.detect_faces = DetectFaces()
        self.image_directory = './Pictures/People'  # Chemin vers les images de test
        self.image_paths = [os.path.join(self.image_directory, img) for img in os.listdir(self.image_directory) if img.endswith('.jpg')]

    def test_face_detection(self):
        """
        Test the face detection on real images.
        """
        for img_path in self.image_paths:
            # Lire l'image
            image = cv2.imread(img_path)
            self.assertIsNotNone(image, f"Image {img_path} could not be read")

            # Détecter les visages (ROIs)
            rois = self.detect_faces.get_rois(image)
            embeddings = self.detect_faces.get_embeddings(image)

            # Vérifier que le nombre de ROIs correspond au nombre d'embeddings
            self.assertEqual(len(rois), len(embeddings), "The number of detected faces should match the number of embeddings")

            # Vérifier que chaque embedding a la bonne dimension (généralement 512 pour InsightFace)
            for emb in embeddings:
                self.assertEqual(len(emb), 512, "Each embedding should have 512 dimensions")
    
    def test_no_faces(self):
        """
        Test that no faces are detected on an image without faces (use a blank image or a specific test image).
        """
        # Créer une image blanche (aucun visage)
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Détecter les visages
        rois = self.detect_faces.get_rois(blank_image)
        embeddings = self.detect_faces.get_embeddings(blank_image)

        # Vérifier qu'aucun visage ou embedding n'est trouvé
        self.assertEqual(len(rois), 0, "No faces should be detected in a blank image")
        self.assertEqual(len(embeddings), 0, "No embeddings should be detected in a blank image")

if __name__ == '__main__':
    unittest.main()
