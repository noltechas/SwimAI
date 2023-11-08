import os
import cv2
import xml.etree.ElementTree as ET

def crop_and_save_images(data_dir, annotation_dir, output_dir, padding_factor=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for annotation_file in os.listdir(annotation_dir):
        # Exclude augmented data
        if "_vertical" in annotation_file or "_horizontal" in annotation_file or "_both" in annotation_file:
            continue

        tree = ET.parse(os.path.join(annotation_dir, annotation_file))
        root = tree.getroot()

        filename = root.find('filename').text
        img_path = os.path.join(data_dir, filename)

        img = cv2.imread(img_path)

        bndbox = root.find('object').find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Calculate padding for width and height
        width_padding = int((xmax - xmin) * padding_factor)
        height_padding = int((ymax - ymin) * padding_factor)

        # Adjust bounding box with padding
        xmin = max(0, xmin - width_padding)
        ymin = max(0, ymin - height_padding)
        xmax = min(img.shape[1], xmax + width_padding)
        ymax = min(img.shape[0], ymax + height_padding)

        cropped_img = img[ymin:ymax, xmin:xmax]

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped_img)

data_dir = 'processed_frames'
annotation_dir = 'processed_frames_labels'
output_dir = 'cropped_frames'

crop_and_save_images(data_dir, annotation_dir, output_dir)
