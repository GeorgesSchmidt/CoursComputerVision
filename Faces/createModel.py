import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import argparse  # Importing argparse for command-line argument parsing

from Faces.createData import CreateDatas

class CreateModel(CreateDatas):
    def __init__(self, directory, title) -> None:
        """
        Initialize the CreateModel class by loading the data, calculating the SVM model, 
        and saving the model to a file.
        
        Args:
            directory (str): Path to the directory containing the data.
            title (str): Name of the file where the model will be saved.
        """
        super().__init__(directory)
        
        self.title = title
        self.calculate_model()
        self.save_model()
                
    def calculate_model(self):
        """
        Calculate the SVM model using the data.
        - Splits the data into training and testing sets.
        - Trains the SVM model with a linear kernel.
        - Prints the accuracy scores for both training and testing sets.
        
        Note: Accuracy score is the percentage of correct predictions compared to the total number of predictions.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle=True, random_state=17)
        print('Data sizes: X_train:', len(X_train), 'X_test:', len(X_test), 'y_train:', len(y_train), 'y_test:', len(y_test))
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)
        ypreds_train = self.model.predict(X_train)
        ypreds_test = self.model.predict(X_test)
        print('Training accuracy score:', accuracy_score(y_train, ypreds_train))
        print('Testing accuracy score:', accuracy_score(y_test, ypreds_test))
        
    def save_model(self):
        """
        Save the trained SVM model to a file using joblib.
        """
        joblib.dump(self.model, self.title)
        print(f'Model {self.title} saved successfully!')
        
if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Train and save an SVM model using image data.")

    # Add arguments for the data directory and model output title
    parser.add_argument('directory', type=str, help="Path to the directory containing the image data.")
    parser.add_argument('title', type=str, help="Name of the file where the trained model will be saved.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the CreateModel class with parsed arguments
    CreateModel(args.directory, args.title)
