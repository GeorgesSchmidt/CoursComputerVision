import cv2
import numpy as np
import os
from Faces.detectFaces import DetectFaces

class CreateDatas(DetectFaces):
    def __init__(self, directory) -> None:
        """
        Initializes the CreateDatas class by creating datasets for two categories: 'People' and 'King'. 
        It stores image embeddings in X and labels in y. 
        'People' category is labeled as 0 and 'King' category as 1.
        
        Args:
            directory (str): The root directory containing 'People' and 'King' subdirectories with images.
        """
        super().__init__()
    
        self.X, self.y = [], []
        no_king_emb = self.get_datas(os.path.join(directory, 'People'))
        for emb in no_king_emb:
            self.X.append(emb)
            self.y.append(0)
            
        king_emb = self.get_datas(os.path.join(directory, 'King'))
        for emb in king_emb:
            self.X.append(emb)
            self.y.append(1)
            
        print('people embeddings =', len(self.X), 'king embeddings', len(self.y))
        
        
    def get_datas(self, path):
        """
        Retrieves the embeddings of all images in the specified directory.

        Args:
            path (str): Path to the directory containing image files.

        Returns:
            list: A list of image embeddings where each embedding is an array representing facial features.
        """
        paths = os.listdir(path)
        results = []
        for p in paths:
            p = os.path.join(path, p)
            image = cv2.imread(p)
            if image is not None:
                embeddings = self.get_embeddings(image)
                if len(embeddings) > 0:
                    results.append(embeddings[0])
        return results
            
            
        
        
def main(direct):
    """
    Main function to initialize the CreateDatas class and process image embeddings.

    Args:
        direct (str): The directory containing 'People' and 'King' subdirectories with images.
    """
    datas = CreateDatas(direct)
    
    
if __name__=='__main__':
    directory = './Pictures'
    main(directory)
