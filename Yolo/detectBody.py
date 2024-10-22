import cv2
import os
from ultralytics import YOLO
import supervision as sv

class DetectYOLO:
    def __init__(self) -> None:
        """
        Initializes the DetectYOLO class with a YOLOv8 segmentation model and annotators for 
        labeling and mask detection.
        """
        self.model = YOLO('yolov8s-seg.pt')  # Load YOLOv8 segmentation model
        self.label_annotator = sv.LabelAnnotator()  # Annotator for adding labels to the image
        self.mask_annotator = sv.MaskAnnotator(opacity=0.7)  # Annotator for visualizing masks with given opacity
        
    def detect_body(self, img):
        """
        Performs object detection on the input image using the YOLO model and prints the results.

        Args:
            img (ndarray): The input image for object detection.
        """
        results = self.model(img, verbose=False)[0]
        print('results', results)
        
        
    def get_boxes(self, img):
        """
        Detects objects in the image, retrieves their bounding boxes, and adds annotations 
        with class labels to the image.
        
        Args:
            img (ndarray): The input image for detection.
        
        Returns:
            annotated_image (ndarray): The image with annotated bounding boxes and labels.
        """
        results = self.model(img)[0]  # Run the model on the input image
        detections = sv.Detections.from_ultralytics(results)  # Convert YOLO results to Detections format
        bounding_box_annotator = sv.BoundingBoxAnnotator()  # Create bounding box annotator
        label_annotator = sv.LabelAnnotator()  # Create label annotator

        # Generate labels based on detected object class IDs
        labels = [
            self.model.model.names[class_id]
            for class_id in detections.class_id
        ]

        # Annotate bounding boxes and labels onto the image
        annotated_image = bounding_box_annotator.annotate(
            scene=img, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image
    
    def get_masks(self, img):
        """
        Detects objects in the image and adds annotations with segmentation masks.

        Args:
            img (ndarray): The input image for mask detection.

        Returns:
            annotated_frame (ndarray): The image with annotated segmentation masks.
        """
        results = self.model(img)[0]  # Run the model on the input image
        detections = sv.Detections.from_ultralytics(results)  # Convert YOLO results to Detections format
        
        # Annotate masks onto the image
        annotated_frame = self.mask_annotator.annotate(
            scene=img.copy(),
            detections=detections
        )
        return annotated_frame
        
def main(img_path):
    """
    Main function to load an image, detect objects using YOLO, and display the results
    with both bounding box and mask annotations.

    Args:
        img_path (str): The path to the image file.
    """
    image = cv2.imread(img_path)  # Load the image from the given path
    detect = DetectYOLO()  # Create an instance of DetectYOLO

    # Get the image with bounding boxes annotated
    img_box = detect.get_boxes(image)

    # Get the image with segmentation masks annotated
    img_mask = detect.get_masks(image)

    # Display the results side by side (bounding box + mask annotations)
    cv2.imshow('', cv2.hconcat([img_box, img_mask]))
    cv2.waitKey(0)

    
if __name__=='__main__':
    image_directory = './Pictures/People'  # Directory containing the images
    paths = os.listdir(image_directory)  # Get the list of image paths
    
    # Process each image in the directory
    for path in paths:
        p = os.path.join(image_directory, path)
        main(p)  # Run the main function on each image
