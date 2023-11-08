import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
import numpy as np
import cv2
from keras import backend as K
from load_data import X_test, y_test

MODEL_PATH = 'saved_model.h5'


def predict_image(model, image_path, target_size=(480, 640)):
    # Load the image in grayscale
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = img_original.shape
    
    # Resize the image to the expected input shape for prediction
    img_resized = cv2.resize(img_original, target_size)
    img_resized_normalized = img_resized / 255.0
    img_resized_normalized = np.expand_dims(np.expand_dims(img_resized_normalized, axis=-1), axis=0)  # Add channel and batch dimensions

    # Predict bounding box
    prediction = model.predict(img_resized_normalized)[0]
    
    # Convert normalized coordinates back to resized image dimensions
    box_resized = [
        int(prediction[0] * target_size[1]),
        int(prediction[1] * target_size[0]),
        int(prediction[2] * target_size[1]),
        int(prediction[3] * target_size[0])
    ]

    # Scale the bounding box coordinates to the original image dimensions
    box_original = [
        int(box_resized[0] * (original_width / target_size[1])),
        int(box_resized[1] * (original_height / target_size[0])),
        int(box_resized[2] * (original_width / target_size[1])),
        int(box_resized[3] * (original_height / target_size[0]))
    ]

    # Draw bounding box on the original image
    img_original_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_original_bgr, (box_original[0], box_original[1]), (box_original[2], box_original[3]), (0, 255, 0), 2)

    # Display the image with bounding box
    cv2.imshow('Prediction', img_original_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_video(model, video_path, target_size=(480, 640), output_path='output_video.mp4'):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames

        # Convert the frame to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize the frame for prediction
        img_resized = cv2.resize(img_gray, target_size)
        img_resized_normalized = img_resized / 255.0
        img_resized_normalized = np.expand_dims(np.expand_dims(img_resized_normalized, axis=-1), axis=0)

        # Predict bounding box
        prediction = model.predict(img_resized_normalized)[0]
        
        # Convert normalized coordinates back to resized frame dimensions
        box_resized = [
            int(prediction[0] * target_size[1]),
            int(prediction[1] * target_size[0]),
            int(prediction[2] * target_size[1]),
            int(prediction[3] * target_size[0])
        ]

        # Scale the bounding box coordinates to the original frame dimensions
        box_original = [
            int(box_resized[0] * (original_width / target_size[1])),
            int(box_resized[1] * (original_height / target_size[0])),
            int(box_resized[2] * (original_width / target_size[1])),
            int(box_resized[3] * (original_height / target_size[0]))
        ]

        # Draw bounding box on the original frame
        cv2.rectangle(frame, (box_original[0], box_original[1]), (box_original[2], box_original[3]), (0, 255, 0), 2)

        # Write the frame with bounding box to the output video
        out.write(frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Call the function
video_path = 'trimmedGX010008.mov'
predict_video(model, video_path)
