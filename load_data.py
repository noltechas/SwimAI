import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_data(data_dir, annotation_dir, img_size=(480, 640)):
    images = []
    boxes = []
    filenames = []  # Store the filenames

    for annotation_file in sorted(os.listdir(annotation_dir)):
        tree = ET.parse(os.path.join(annotation_dir, annotation_file))
        root = tree.getroot()

        filename = root.find('filename').text
        img_path = os.path.join(data_dir, filename)
        
        # Store the filename
        filenames.append(filename)

    for annotation_file in os.listdir(annotation_dir):
        tree = ET.parse(os.path.join(annotation_dir, annotation_file))
        root = tree.getroot()

        filename = root.find('filename').text
        img_path = os.path.join(data_dir, filename)
        
        # Read the image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was loaded correctly
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Get the original dimensions
        original_height, original_width = img.shape
        
        # Resize the image to the specified size
        img = cv2.resize(img, img_size)
        
        # Expand dimensions to make it a single-channel image
        img = np.expand_dims(img, axis=-1)
        
        img = img / 255.0  # Normalize to [0, 1]

        bndbox = root.find('object').find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Adjust bounding box coordinates based on the new image size
        xmin = int((xmin / original_width) * img_size[0])
        ymin = int((ymin / original_height) * img_size[1])
        xmax = int((xmax / original_width) * img_size[0])
        ymax = int((ymax / original_height) * img_size[1])

        # Normalize bounding box coordinates
        box = [xmin/img_size[0], ymin/img_size[1], xmax/img_size[0], ymax/img_size[1]]

        images.append(img)
        boxes.append(box)

    return np.array(images), np.array(boxes), filenames

data_dir = 'processed_frames'
annotation_dir = 'processed_frames_labels'
X, y, filenames = load_data(data_dir, annotation_dir)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
