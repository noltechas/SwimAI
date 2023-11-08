import cv2
import numpy as np
import tensorflow as tf

# Load the models
swimmer_model = tf.keras.models.load_model('swimmer_detection_model.h5')
pose_model = tf.keras.models.load_model('pose_model.h5')

def predict_swimmer_location(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to match the expected input shape of the model
    resized_frame = cv2.resize(gray_frame, (480, 640))
    
    # Normalize and expand dimensions
    input_img = np.expand_dims(resized_frame, axis=[0, -1]) / 255.0
    
    # Predict the bounding box
    box = swimmer_model.predict(input_img)[0]
    
    # Convert the normalized bounding box coordinates back to original frame dimensions
    xmin = int(box[0] * frame.shape[1])
    ymin = int(box[1] * frame.shape[0])
    xmax = int(box[2] * frame.shape[1])
    ymax = int(box[3] * frame.shape[0])
    
    return xmin, ymin, xmax, ymax


def predict_body_parts(roi):
    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Resize the ROI to match the expected input shape of the pose model
    resized_roi = cv2.resize(gray_roi, (480, 640))
    
    # Normalize and expand dimensions
    normalized_roi = resized_roi / 255.0
    input_roi = np.expand_dims(normalized_roi, axis=[0, -1])
    
    # Predict the locations of the hands and feet
    predictions = pose_model.predict(input_roi)[0]
    
    # Assuming the model returns predictions in the format [x1, y1, x2, y2, x3, y3, x4, y4]
    hand1 = (predictions[0], predictions[1])
    hand2 = (predictions[2], predictions[3])
    foot1 = (predictions[4], predictions[5])
    foot2 = (predictions[6], predictions[7])
    
    return hand1, hand2, foot1, foot2


# Open the video
cap = cv2.VideoCapture('trimmedGX010008.mov')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict the swimmer's location
    xmin, ymin, xmax, ymax = predict_swimmer_location(frame)

    # Calculate the width and height of the bounding box
    box_width = xmax - xmin
    box_height = ymax - ymin

    # Apply 10% padding to the bounding box dimensions
    xmin = max(0, xmin - int(0.1 * box_width))
    ymin = max(0, ymin - int(0.1 * box_height))
    xmax = min(frame.shape[1], xmax + int(0.1 * box_width))
    ymax = min(frame.shape[0], ymax + int(0.1 * box_height))

    # Draw the bounding box around the swimmer
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue color for the bounding box

    # Extract the ROI
    roi = frame[ymin:ymax, xmin:xmax]

    # Predict the body parts within the ROI
    hand1, hand2, foot1, foot2 = predict_body_parts(roi)

    # Draw the predictions on the original frame
    for (x, y) in [hand1, hand2]:
        if x > 0 and y > 0:  # Check if the coordinates are not zeros
            x = int(x * (xmax - xmin) + xmin)
            y = int(y * (ymax - ymin) + ymin)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green color for hands

    for (x, y) in [foot1, foot2]:
        if x > 0 and y > 0:  # Check if the coordinates are not zeros
            x = int(x * (xmax - xmin) + xmin)
            y = int(y * (ymax - ymin) + ymin)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red color for feet

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()