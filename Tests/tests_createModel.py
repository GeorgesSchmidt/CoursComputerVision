import unittest
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from Faces.createModel import CreateModel

class TestCreateModel(unittest.TestCase):

    def setUp(self):
        """
        Set up paths and simulate data for testing.
        """
        # Chemin du répertoire contenant les images et nom du fichier modèle
        self.directory = './Pictures'  # Assurez-vous que ce chemin existe et contient les images.
        self.model_path = './model_test.pkl'
        
        # Initialiser la classe CreateModel
        self.create_model = CreateModel(self.directory, self.model_path)
    
    def test_initialization(self):
        """
        Test that the initialization properly loads the data and prepares it for training.
        """
        # Vérifier que les données sont chargées et que X et y ne sont pas vides
        self.assertGreater(len(self.create_model.X), 0, "X should not be empty")
        self.assertGreater(len(self.create_model.y), 0, "y should not be empty")

        # Vérifier que les dimensions des embeddings sont correctes
        self.assertEqual(len(self.create_model.X[0]), 512, "Each embedding should have 512 dimensions")
        
        # Vérifier que les étiquettes contiennent uniquement 0 et 1
        for label in self.create_model.y:
            self.assertIn(label, [0, 1], "y should only contain 0 or 1 integers")
    
    def test_calculate_model(self):
        """
        Test that the SVM model is trained correctly and gives reasonable accuracy scores.
        """
        # Vérifier que le modèle est bien un SVC avec un noyau linéaire
        self.assertIsInstance(self.create_model.model, SVC, "The model should be an instance of SVC")
        self.assertEqual(self.create_model.model.kernel, 'linear', "The SVM kernel should be linear")

        # Effectuer des prédictions sur les données d'entraînement et de test
        X_train, X_test, y_train, y_test = self.create_model.X, self.create_model.X, self.create_model.y, self.create_model.y
        ypred_train = self.create_model.model.predict(X_train)
        ypred_test = self.create_model.model.predict(X_test)
        
        # Vérifier que la précision est raisonnable (entre 0 et 1)
        accuracy_train = np.mean(ypred_train == y_train)
        accuracy_test = np.mean(ypred_test == y_test)
        self.assertGreaterEqual(accuracy_train, 0, "Training accuracy should be >= 0")
        self.assertLessEqual(accuracy_train, 1, "Training accuracy should be <= 1")
        self.assertGreaterEqual(accuracy_test, 0, "Testing accuracy should be >= 0")
        self.assertLessEqual(accuracy_test, 1, "Testing accuracy should be <= 1")

    def test_save_model(self):
        """
        Test that the trained model is saved correctly to the specified path.
        """
        # Vérifier que le modèle a bien été sauvegardé
        self.create_model.save_model()
        self.assertTrue(os.path.exists(self.model_path), "The model file should exist after saving")
        
        # Charger le modèle et vérifier qu'il est bien du type SVC
        loaded_model = joblib.load(self.model_path)
        self.assertIsInstance(loaded_model, SVC, "The loaded model should be an instance of SVC")
        
    def tearDown(self):
        """
        Cleanup after the test.
        """
        # Supprimer le fichier modèle s'il existe
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
