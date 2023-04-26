#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os

# Path to the folder containing the images
path = 'C:/input/training\ww'

# Standard size for the resized images
size = (128, 128)

# Loop through all the files in the folder
for filename in os.listdir(path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image using PIL
        img = Image.open(os.path.join(path, filename))
        # Convert the image to grayscale
        grayscale_img = img.convert("L")
        # Resize the image to the standard size
        resized_img = grayscale_img.resize(size)
        # Save the resized image, overwriting the original
        resized_img.save(os.path.join(path, filename))


# In[3]:


from PIL import Image
import os

# Path to the folder containing the images
path = 'C:\input\character segmentation'

# Standard size for the resized images
size = (128, 128)

# Loop through all the files in the folder
for filename in os.listdir(path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(f"Processing image {filename}...")
        # Open the image using PIL
        img = Image.open(os.path.join(path, filename))
        # Convert the image to grayscale
        grayscale_img = img.convert("L")
        # Resize the image to the standard size
        resized_img = grayscale_img.resize(size)
        # Save the resized image, overwriting the original
        resized_img.save(os.path.join(path, filename))
        print(f"Image {filename} processed and saved.")


# In[ ]:





# In[ ]:




