#!/usr/bin/env python
# coding: utf-8

# In[3]:


from rembg import remove
from PIL import Image
import os

#Set the working directory to the folder containing the images
os.chdir("")

#Create a subdirectory
if not os.path.exists('Background Removed'):
    os.mkdir('Background Removed')

#Loop over every file in the folder
for image_file in os.listdir():
    #Skip non-image files
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    #Define the input and output paths for the current image
    input_path = image_file
    output_path = os.path.join('Background Removed', image_file)
    #Open the image and remove the background
    input_image = Image.open(input_path)
    output_image = remove(input_image, bgcolor=(255,255,255,255))

    #Convert the resulting image to RGB format
    output_image = output_image.convert("RGB")
    #output_image = output_image.point(lambda p: p * 1.5) #Changes background color
    # save the resulting image in the correct subfolder
    output_image.save(output_path)


# In[ ]:




