# -*- coding: utf-8 -*-
"""Image Preparation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aGPaz0fMohXUPsqE_14XCjvGqiMAwqQB
"""

#This code uses the dlib package to detect faces in images and rotate them so that they are completely vertical.
#As I was using multiple sets of images, I decided it would be faster to go through every subfolder in a folder (each containing an image set)
#Rather than manually input the folders one-by-one
#The rotated images are saved in new folders

#Load the packages
import dlib
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import rotate
from math import atan2, degrees

#Load the dlib predictor to identify 68 landmarks on faces
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/content/gdrive/MyDrive/shape_predictor_68_face_landmarks.dat')

# Base directory containing all subfolders
base_folder_path = ''

#Can be modified if you want to skip any folders, but probably easier just to move files
start_subfolder_index = 0

# Get the list of subfolders
subfolders = [folder for folder in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, folder))]

# Iterate through all subfolders starting from the specified index
for subfolder in subfolders[start_subfolder_index:]:
    subfolder_path = os.path.join(base_folder_path, subfolder)
    print(subfolder_path) #You can check if any were missed
    #The rotated images will be saved in new subfolders
    rotated_subfolder_path = os.path.join(base_folder_path, f"{subfolder}_rotated")
    os.makedirs(rotated_subfolder_path, exist_ok=True)

    # Iterate through all images in the subfolder
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.gif'):
            # Load the image
            image_path = os.path.join(subfolder_path, filename)
            image = io.imread(image_path)

            # Detect faces in the image
            faces = detector(image, 1)

            # If no faces are detected, save the original image and print an alert
            if not faces:
                output_path = os.path.join(rotated_subfolder_path, filename)
                io.imsave(output_path, image)
                #print(f"Alert: No face detected in {filename} from {subfolder}. Original image saved.")
            else:
                # Iterate over the detected faces
                #In my case, there should have been just one, but this allows a margin of error
                for face in faces:
                    # Find the facial landmarks
                    landmarks = predictor(image, face)
                    #This code is somewhat opaque, but in essence it's finding the difference in degrees between the eyes, then rotating the image by that many degrees
                    eye_left = (landmarks.part(36).x, landmarks.part(36).y)
                    eye_right = (landmarks.part(45).x, landmarks.part(45).y)
                    dx = eye_right[0] - eye_left[0]
                    dy = eye_right[1] - eye_left[1]
                    angle = atan2(dy, dx) * 180. / np.pi  # Convert from radians to degrees

                    # Rotate image
                    rotated_img = rotate(image, angle, mode='edge')
                    #Converting to uint8 saves memory
                    rotated_img_uint8 = np.clip(rotated_img * 255, 0, 255).astype(np.uint8)

                    # Save the rotated image in the output directory
                    output_path = os.path.join(rotated_subfolder_path, filename)
                    io.imsave(output_path, rotated_img_uint8)
#This code is easily modifiable to save the images in a new folder, but in my case I wanted to use further image preparation techniques on both sets, so I kept them together.

#MTCNN is a deep learning method for detecting faces. The mtcnn function from facenet-pytorch allows images to be cropped with variable margins around faces.

import os
from PIL import Image
from facenet_pytorch import MTCNN

# Set the path for the input folder
input_folder = ''

# Create the output folder path as a subfolder within the input folder
output_folder = os.path.join(input_folder, "_mtcnn")

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create an instance of MTCNN
mtcnn = MTCNN(select_largest=True, image_size=224, post_process=True)

# Iterate through each image file in the input folder
for file_name in os.listdir(input_folder):
    # Open the image
    image_path = os.path.join(input_folder, file_name)
    if not os.path.isfile(image_path) or not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        continue

    try:
        img = Image.open(image_path)

        # Check if the image is grayscale
        #This was necessary for training my neural networks later.
        if img.mode == 'L':
            # Convert grayscale image to RGB
            img_rgb = img.convert('RGB')
            img = img_rgb

        output_path = os.path.join(output_folder, file_name)
        img_cropped = mtcnn(img, save_path=output_path)

    except Exception as e:
        print(f"Error processing image: {file_name}")
        print(f"Error message: {str(e)}")

print("Image processing completed.")