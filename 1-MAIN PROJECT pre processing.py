#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("C:\input\malayalam\hd-2.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove noise from the image using Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#blurred = cv2.medianBlur(gray, 5)

# Apply Otsu's thresholding to the image
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# Perform morphological operations to remove noise and thin the image
kernel = np.ones((3,3), np.uint8)
skeleton1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
skeleton2 = cv2.morphologyEx(skeleton1, cv2.MORPH_OPEN, kernel)
skeleton = cv2.morphologyEx(skeleton2, cv2.MORPH_CLOSE, kernel)



# Save the processed image
cv2.imwrite("C:\input\malayalam\cp10.png", binary)

plt.figure(figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.imshow(image,cmap=plt.cm.gray)
plt.title('input image')
plt.axis('off')



plt.subplot(2, 2, 2)
plt.imshow(binary, cmap=plt.cm.gray)
plt.title('output')
plt.axis('off')



plt.show()

if cv2.waitKey(0) & 0xff == 27:
 cv2.destroyAllWindows()


# In[ ]:





# In[3]:


#character segmentation
#fill=cv2.GaussianBlur(img_di, (7, 7), 0)
#ret,thresh1 = cv2.threshold(fill ,127,255,cv2.THRESH_BINARY_INV)
canny = cv2.Canny(skeleton, 120, 255, 1)
dilate = cv2.dilate(canny, None, iterations=1)


cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
#print(cnts)
cnts = cnts[0] 
sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[1] * image.shape[1] )

orig = skeleton.copy()
plt.imshow(orig,'gray')
i = 0
for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv2.contourArea(cnt) < 500):
        continue

    # Filtered countours are detected
    x,y,w,h = cv2.boundingRect(cnt)
    
    # Taking ROI of the cotour
    roi = skeleton[y:y+h, x:x+w]
    # Mark them on the image if you want
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),3)
    # Save your contours or characters
    cv2.imwrite("C:\input\malayalam\e1char" + str(i) + ".png", roi)
    
    i = i + 1   

cv2.imwrite("C:\input\malayalam\e1 character.jpg",orig)
plt.figure(figsize=(15,15))
plt.title("Character Segmetation")
plt.imshow(orig,'gray')
cnts.clear()


# In[ ]:





# In[ ]:


from skimage.feature import hog
from skimage import io
import os
import numpy as np

# Define the HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)
visualize = False

# Define the input and output paths
input_folder_path = "C:\input\MalayalamHandwrittenDataset\character_3454"
output_feature_path = "C:/output/features/ൾ"
output_label_path = "C:/output/labels/ൾ"

# Create the output folders if they do not exist
os.makedirs(output_feature_path, exist_ok=True)
os.makedirs(output_label_path, exist_ok=True)

# Initialize empty lists to store the features and labels of each image in the folder
features = []
labels = []

# Loop over all images in the folder
for i, image_name in enumerate(os.listdir(input_folder_path)):
    image_path = os.path.join(input_folder_path, image_name)
    
    # Load the image
    image = io.imread(image_path, as_gray=True)
    
    # Extract the HOG features from the image
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=visualize)
    
    # Append the features to the list
    features.append(hog_features)
    
    # Create a corresponding label for the image
    label = "ൾ"
    
    # Append the label to the list
    labels.append(label)
    
    # Print the progress
    print(f"Processed image {i+1}/{len(os.listdir(input_folder_path))}")

# Convert the lists of features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Save the features and labels to files in the output folders
np.save(os.path.join(output_feature_path, "features.npy"), features)
np.save(os.path.join(output_label_path, "labels.npy"), labels)

# Print the total number of images and features
print(f"Processed {len(os.listdir(input_folder_path))} images, extracted {features.shape[1]} features each")


# In[3]:


import cv2
import pytesseract

# Load the image
image = cv2.imread("C:\input\malayalam\hd-1-1.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR using Tesseract
text = pytesseract.image_to_string(gray, lang='mal')

# Print the recognized text
print(text)


# In[ ]:




