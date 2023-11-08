import os
import cv2
import xml.etree.ElementTree as ET
import random

def save_augmented_image_and_label(img, annotation_path, flip_type, save_dir, save_annotation_dir, base_name):
    # Flip the image
    flipped_img = cv2.flip(img, flip_type)
    
    # Save the flipped image
    flip_suffix = {0: "_vertical", 1: "_horizontal", -1: "_both"}[flip_type]
    flipped_img_name = base_name + flip_suffix + ".jpg"
    flipped_img_path = os.path.join(save_dir, flipped_img_name)
    cv2.imwrite(flipped_img_path, flipped_img)
    
    # Parse the XML annotation
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    # Iterate over each object in the annotation
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        
        # Get the bounding box coordinates
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Flip the bounding box coordinates based on flip type
        height, width = img.shape[:2]
        if flip_type in [1, -1]:  # Horizontal flip or both
            xmin, xmax = width - xmax, width - xmin
        if flip_type in [0, -1]:  # Vertical flip or both
            ymin, ymax = height - ymax, height - ymin
        
        # Update the XML annotation with the flipped coordinates
        bndbox.find('xmin').text = str(xmin)
        bndbox.find('ymin').text = str(ymin)
        bndbox.find('xmax').text = str(xmax)
        bndbox.find('ymax').text = str(ymax)
    
    # Update the filename in the XML annotation
    root.find('filename').text = flipped_img_name

    # Save the updated XML annotation
    flipped_annotation_name = base_name + flip_suffix + ".xml"
    flipped_annotation_path = os.path.join(save_annotation_dir, flipped_annotation_name)
    tree.write(flipped_annotation_path)

# Directory paths
data_dir = 'cropped_frames'
annotation_dir = 'cropped_frames_labels'

# Iterate over each image file and save the augmented images and their corresponding labels
for img_file in os.listdir(data_dir):
    base_name, ext = os.path.splitext(img_file)
    
    # Check if the frame has already been augmented
    if "_vertical" in base_name or "_horizontal" in base_name or "_both" in base_name:
        continue

    # Check if the corresponding XML annotation file exists
    annotation_file = base_name + ".xml"
    annotation_path = os.path.join(annotation_dir, annotation_file)
    if not os.path.exists(annotation_path):
        continue

    img_path = os.path.join(data_dir, img_file)
    img = cv2.imread(img_path)
    
    for flip_type in [0, 1, -1]:
        save_augmented_image_and_label(img, annotation_path, flip_type, data_dir, annotation_dir, base_name)
