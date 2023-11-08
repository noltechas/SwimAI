import cv2, os
import numpy as np


def preprocess_image(img):
    # No need to read the image again, as we're passing the image directly

    # Apply Bilateral Filter for noise reduction
    img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)

    # Extract the Value channel and apply CLAHE
    value_channel = img_hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_value = clahe.apply(value_channel)

    # Merge the enhanced Value channel back into the HSV image
    img_hsv[:, :, 2] = enhanced_value
    img_enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Apply morphological closing
    kernel = np.ones((5, 5), np.uint8)
    img_closed = cv2.morphologyEx(img_enhanced, cv2.MORPH_CLOSE, kernel)

    return img


def sobel_edge_detection(img_gray, threshold_value=100):
    # Apply Sobel operator in the X and Y directions
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradients
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Convert to 8-bit image
    sobel_output = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply thresholding to keep only strong edges
    _, sobel_thresholded = cv2.threshold(sobel_output, threshold_value, 255, cv2.THRESH_BINARY)

    # Ensure the output is a single-channel grayscale image
    if len(sobel_thresholded.shape) == 3:
        sobel_thresholded = cv2.cvtColor(sobel_thresholded, cv2.COLOR_BGR2GRAY)

    return sobel_thresholded

def process_video(video_path, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")

    frame_num = 0
    for _ in range(total_frames):
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if video is over or frame couldn't be read
        if not ret:
            print(f"Stopped at frame {frame_num} due to read error.")
            break

        # Apply preprocessing and edge detection
        preprocessed_frame = preprocess_image(frame)
        edge_detected_frame = sobel_edge_detection(preprocessed_frame)

        # Save the processed frame
        output_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
        cv2.imwrite(output_path, preprocessed_frame)

        frame_num += 1
        if frame_num % 100 == 0:
            print("Processed " + str(frame_num) + " frames")

    # Release the video capture object
    cap.release()

    print(f"Processed {frame_num} frames and saved to {output_dir}")


def frames_to_video(input_dir, output_video_path, fps=30):
    # Get the list of frames from the input directory
    frames = [os.path.join(input_dir, frame) for frame in os.listdir(input_dir) if frame.endswith('.jpg')]

    # Sort frames in numerical order
    frames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.jpg')[0]))

    # Check if there are any frames to process
    if len(frames) == 0:
        print("No frames in directory", input_dir)
        return

    # Find out the frame size from the first image
    frame_size = cv2.imread(frames[0]).shape[:2]
    frame_size = (frame_size[1], frame_size[0])  # width, height

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for frame in frames:
        img = cv2.imread(frame)
        out.write(img)

    out.release()
    print(f"Video {output_video_path} created successfully")


# Example usage
video_path = "breaststroke.mp4"
output_directory = "processed_frames"
output_video_path = "output_video.mp4"

process_video(video_path, output_directory)
#frames_to_video(output_directory, output_video_path)