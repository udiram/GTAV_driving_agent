import cv2
import os

# Get all of the PNG images in the directory
image_folder = 'data/training_1000/'
images = [img for img in os.listdir(image_folder) if img.endswith('.png')]

# Sort the images by name
images.sort()

# Set the frame rate and codec for the video
frame_rate = 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Get the size of the first image
img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = img.shape

# Create the video writer
video = cv2.VideoWriter('video.avi', fourcc, frame_rate, (width,height))

# Add each image to the video
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the video writer
video.release()