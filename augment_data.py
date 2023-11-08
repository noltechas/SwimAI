import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
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
    bndbox = root.find('object').find('bndbox')
    
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
    
    # Save the updated XML annotation
    flipped_annotation_name = base_name + flip_suffix + ".xml"
    flipped_annotation_path = os.path.join(save_annotation_dir, flipped_annotation_name)
    tree.write(flipped_annotation_path)

    # Update the filename in the XML annotation
    root.find('filename').text = flipped_img_name

    # Save the updated XML annotation
    flipped_annotation_name = base_name + flip_suffix + ".xml"
    flipped_annotation_path = os.path.join(save_annotation_dir, flipped_annotation_name)
    tree.write(flipped_annotation_path)

# Directory paths
data_dir = 'new_frames'
annotation_dir = 'new_labels'

# Iterate over each annotation file and save the augmented images and their corresponding labels
for annotation_file in os.listdir(annotation_dir):
    annotation_path = os.path.join(annotation_dir, annotation_file)
    base_name = os.path.splitext(annotation_file)[0]
    img_name = base_name + ".jpg"
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path)
    
    for flip_type in [0, 1, -1]:
        save_augmented_image_and_label(img, annotation_path, flip_type, data_dir, annotation_dir, base_name)

# Display a random frame with its bounding box and its corresponding augmented frames
def display_image_with_box(img_path, annotation_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bndbox = root.find('object').find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    plt.imshow(img, cmap='gray')
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red'))
    plt.show()

random_file = random.choice(os.listdir(data_dir))
base_name = 'breaststroke3_frame'
display_image_with_box(os.path.join(data_dir, random_file), os.path.join(annotation_dir, base_name + ".xml"))
for flip_suffix in ["_horizontal", "_vertical", "_both"]:
    display_image_with_box(os.path.join(data_dir, base_name + flip_suffix + ".jpg"), os.path.join(annotation_dir, base_name + flip_suffix + ".xml"))