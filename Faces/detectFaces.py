import logging
import cv2
import numpy as np
import os

from insightface.app import FaceAnalysis



class DetectFaces:
    def __init__(self) -> None:
        """
        Initializes the DetectFaces class by setting up the FaceAnalysis tool.
        Prepares the face detection model with a specific context ID and detection size.
        """
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def get_embeddings(self, frame):
        """
        Get embeddings for the detected faces in the given frame using the FaceAnalysis model.
        
        Args:
            frame (ndarray): The input image frame.
        
        Returns:
            list: A list of embeddings for each detected face in the frame.
        """
        pred = self.app.get(frame)
        emb = []
        for p in pred:
            emb.append(p['embedding'])
        return emb
    
    def get_rois(self, frame):
        """
        Get Regions of Interest (ROIs) for the detected faces in the given frame.
        Each ROI represents the bounding box of a detected face.

        This method is useful for visualizing face detections or inspecting dataset images.
        
        Args:
            frame (ndarray): The input image frame.
        
        Returns:
            list: A list of bounding boxes (ROIs) for each detected face, represented as arrays of coordinates.
        """
        pred = self.app.get(frame)
        rois = []
        for p in pred:
            box = np.array(p['bbox']).astype(int)
            rois.append(box)
        return rois
    
    
def put_texte(frame, texte):
    """
    Overlay text on the given image frame.

    Args:
        frame (ndarray): The image frame on which to put the text.
        texte (str): The text to be displayed.
    """
    p = (10, 20)  # Position of the text
    font = 1  # Font type
    font_scale = 1.2  # Font size
    color = (0, 0, 255)  # Text color (BGR format)
    thick = 2  # Thickness of the text
    cv2.putText(frame, texte, p, font, font_scale, color, thick)
        
def main(img_path):
    """
    Main function to detect faces, draw bounding boxes, and display face embeddings on an image.
    
    Args:
        img_path (str): The path to the image file where faces are to be detected.
    """
    image = cv2.imread(img_path)  # Load image from the given path
    detect = DetectFaces()  # Create an instance of DetectFaces
    img_result = image.copy()  # Copy the image to avoid altering the original

    # Get bounding boxes (ROIs) of detected faces
    boxes = detect.get_rois(image)
    
    # Draw rectangles around each detected face
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img_result, (x, y), (w, h), (0, 255, 0), 1)
    
    # Get face embeddings for each detected face
    embeddings = detect.get_embeddings(image)
    
    # Concatenate original image with the result image showing face detections
    image = cv2.hconcat([image, img_result])

    # Add text displaying the number of faces detected and embeddings found
    texte = f'{len(boxes)} faces detected / {len(embeddings)} embeddings detected'
    put_texte(image, texte)
    
    # Display the result
    cv2.imshow('', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_directory = './Pictures/People'  # Directory containing images
    paths = os.listdir(image_directory)  # List all files in the directory
    
    # Process each image in the directory
    for path in paths:
        p = os.path.join(image_directory, path)
        main(p)  # Call main function for each image
        
