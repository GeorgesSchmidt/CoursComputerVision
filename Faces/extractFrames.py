import cv2
import numpy as np
import os
import joblib
import argparse  # Importing argparse to handle command-line arguments

from Modules.detectFaces import DetectFaces

class GetFrames(DetectFaces):
    """
    Class to process video frames, detect faces, and make predictions using a pre-trained model. 
    This class extends the DetectFaces class and adds functionality to capture and annotate video frames.
    """
    
    def __init__(self, input_path, output_path, model) -> None:
        """
        Initialize the GetFrames class with the paths to the input video, output video, and model.
        
        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to the output video file where processed frames will be saved.
            model (str): Path to the pre-trained machine learning model used for face prediction.
        """
        super().__init__()
        self.pred = False
        if model is not None and input_path is not None:
            self.pred = True
            self.model = joblib.load(model)
            self.create_recorder(output_path)
            
        self.cap = cv2.VideoCapture(input_path)
        
    def create_recorder(self, title):
        """
        Set up the video writer to save processed frames with the same FPS, width, and height as the input video.
        
        Args:
            title (str): The file path where the output video will be saved.
        """
        cap_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        self.out = cv2.VideoWriter(title, fourcc, cap_fps, (cap_w, cap_h))
        
    def read_video(self):
        """
        Read frames from the video, detect faces, make predictions, and save the processed frames to the output file.
        Displays the video while processing, and stops when the video ends or when the 'Esc' key is pressed.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()  # Read a frame from the video
            if ret:
                if self.pred:
                    self.draw_image(frame)
                    self.out.write(frame)
            
                cv2.imshow('', frame)  # Display the frame in an OpenCV window
                key = cv2.waitKey(1)  # Display each frame for 1ms
                if key == 27:  # Stop video playback if 'Esc' key is pressed
                    break
                
        self.cap.release()  # Release video capture object
        if self.pred:
            self.out.release()  # Release video writer object
        cv2.destroyAllWindows()  # Close any OpenCV windows
        
    def draw_image(self, frame):
        """
        Detect faces in the frame, predict their classes, and annotate the frame with rectangles and labels.
        
        Args:
            frame (numpy.ndarray): The current frame of the video where detections and predictions will be drawn.
        """
        embeddings = self.get_embeddings(frame)
        rois = self.get_rois(frame)
        for roi, emb in zip(rois, embeddings):
            pred = self.get_pred(emb)
            color = (0, 0, 255)  # Default color: red
            if pred == 0:
                color = (0, 0, 0)  # Change color to black if prediction is 0
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (w, h), color, 1)
            texte = f'prediction = {pred}'
            p = (x, y-10)
            self.put_text(frame, texte, p, color)
        
    def get_pred(self, emb):
        """
        Predict whether the face embedding belongs to the "King" class using the pre-trained model.
        
        Args:
            emb (numpy.ndarray): The embedding vector representing the detected face.
            
        Returns:
            pred (int): Prediction result (0 or 1), indicating the class of the detected face.
        """
        emb = np.array(emb).reshape(1, -1)
        [pred] = self.model.predict(emb)
        return pred
    
    def put_text(self, frame, texte, p, color=(0, 0, 0)):
        """
        Add text annotations to a video frame at a specified position.
        
        Args:
            frame (numpy.ndarray): The video frame where the text will be drawn.
            texte (str): The text to display (e.g., prediction result).
            p (tuple): Coordinates (x, y) of where the text should appear on the frame.
            color (tuple): The color of the text (default is black).
        """
        police = cv2.FONT_HERSHEY_SIMPLEX  
        taille_police = 0.5
        epaisseur = 1
        cv2.putText(frame, texte, p, police, taille_police, color, epaisseur, cv2.LINE_AA)

    
def main(input_path, output_path, model):
    """
    Main function to initialize the GetFrames class and start processing the video.
    
    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output video file where processed frames will be saved.
        model (str): Path to the pre-trained machine learning model used for face prediction.
    """
    vid = GetFrames(input_path, output_path, model)
    vid.read_video()

if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Process a video, detect faces, and annotate predictions using a pre-trained model.")
    
    # Add arguments for input video, output video, and model file
    parser.add_argument('input_path', type=str, help="Path to the input video file.")
    parser.add_argument('output_path', type=str, help="Path to the output video file.", default='output.mp4', nargs='?')
    parser.add_argument('model', type=str, help="Path to the pre-trained machine learning model file.", nargs='?', default=None)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.input_path, args.output_path, args.model)
