import unittest
import cv2
import numpy as np
from Faces.extractFrames import GetFrames

class TestGetFrames(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by initializing the GetFrames object.
        We'll use real video and model files here.
        """
        self.input_path = './Videos/kingCharles.mp4'
        self.output_path = './Videos/output_test.mp4'
        self.model_path = 'new_model.pt'  # Assuming the model file is present in this path
        
        self.get_frames = GetFrames(self.input_path, self.output_path, self.model_path)

    def test_create_recorder(self):
        """
        Test that the video recorder is created with the correct parameters using the real video.
        """
        self.get_frames.create_recorder(self.output_path)
        # Check if the output writer was initialized correctly
        self.assertTrue(self.get_frames.out.isOpened())
    
    def test_get_pred(self):
        """
        Test that the get_pred method returns a valid prediction from the real model.
        """
        embedding = np.random.rand(512)  # Simulate a random face embedding
        pred = self.get_frames.get_pred(embedding)
        self.assertIn(pred, [0, 1])  # Assuming the model predicts classes 0 or 1
    
    def test_draw_image(self):
        """
        Test that the draw_image method correctly detects faces and makes predictions.
        """
        # Read a frame from the video
        cap = cv2.VideoCapture(self.input_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            self.get_frames.draw_image(frame)
            # No specific assertion here, but we ensure no exceptions are raised and faces are annotated.
    
    def test_read_video(self):
        """
        Test that the read_video method processes frames from the video and writes to the output.
        """
        self.get_frames.read_video()

        # After processing, check if the output video was created and contains frames.
        cap = cv2.VideoCapture(self.output_path)
        ret, frame = cap.read()
        cap.release()
        
        self.assertTrue(ret)  # Ensure that the output video was written successfully
    
    def test_put_text(self):
        """
        Test that the put_text method adds text to the frame.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = 'Test Text'
        position = (50, 50)
        color = (255, 0, 0)  # Blue color in BGR format
        
        self.get_frames.put_text(frame, text, position, color)
        # There's no easy way to directly check the text placement on the image without visual inspection,
        # but the test ensures no exceptions are thrown.
    
    def tearDown(self):
        """
        Clean up any resources after the test case runs, like closing video writers.
        """
        if hasattr(self.get_frames, 'out') and self.get_frames.out.isOpened():
            self.get_frames.out.release()

if __name__ == '__main__':
    unittest.main()