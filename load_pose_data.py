# load_pose_data.py

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def load_pose_data(data_dir, annotation_dir, img_size=(480, 640)):
    images = []
    all_boxes = []  # List of lists of boxes

    for annotation_file in sorted(os.listdir(annotation_dir)):
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

        hand_boxes = []
        foot_boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
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

            if obj.find('name').text == 'hand':
                hand_boxes.extend(box)
            elif obj.find('name').text == 'foot':
                foot_boxes.extend(box)

        # If there are exactly 2 bounding boxes for hands and 2 for feet, add to the dataset
        if len(hand_boxes) == 8 and len(foot_boxes) == 8:
            boxes = hand_boxes + foot_boxes
            images.append(img)
            all_boxes.append(boxes)

    return np.array(images), np.array(all_boxes)


data_dir = 'cropped_frames'
annotation_dir = 'cropped_frames_labels'
X, y = load_pose_data(data_dir, annotation_dir)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
