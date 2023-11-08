# test_pose_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_PATH = 'pose_model.h5'

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_pose(img_path):
    # Read the image as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded correctly
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    # Get the original dimensions
    original_height, original_width = img.shape
    
    # Resize the image to the expected input shape
    img_resized = cv2.resize(img, (480, 640))
    
    # Normalize the image
    img_normalized = img_resized / 255.0
    
    # Expand dimensions to make it a single-channel image and add batch dimension
    img_input = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
    
    # Predict using the model
    predictions = model.predict(img_input)[0]
    
    # Extract bounding box coordinates for hands and feet
    hand1, hand2, foot1, foot2 = predictions[:4], predictions[4:8], predictions[8:12], predictions[12:]
    
    # Function to scale the bounding boxes back to original dimensions
    def scale_boxes(box):
        return [
            int(box[0] * original_width),
            int(box[1] * original_height),
            int(box[2] * original_width),
            int(box[3] * original_height)
        ]
    
    hand1 = scale_boxes(hand1)
    hand2 = scale_boxes(hand2)
    foot1 = scale_boxes(foot1)
    foot2 = scale_boxes(foot2)
    
    return hand1, hand2, foot1, foot2


def display_predictions(img_path, hand1, hand2, foot1, foot2):
    img = cv2.imread(img_path)
    
    # Function to draw a dot at the center of the bounding box
    def draw_dot(box, color):
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        cv2.circle(img, (center_x, center_y), 5, color, -1)
    
    draw_dot(hand1, (0, 0, 255))  # Red for hand1
    draw_dot(hand2, (0, 0, 255))  # Red for hand2
    draw_dot(foot1, (0, 255, 0))  # Green for foot1
    draw_dot(foot2, (0, 255, 0))  # Green for foot2
    
    cv2.imshow('Predictions', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = 'cropped_frames/breaststroke2_frame_95.jpg'
    hand1, hand2, foot1, foot2 = predict_pose(img_path)
    display_predictions(img_path, hand1, hand2, foot1, foot2)
