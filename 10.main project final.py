#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("C:\input\malayalam\hd-1-1.jpeg")

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


# In[2]:


import cv2
canny = cv2.Canny(skeleton, 120, 255, 1)
dilate = cv2.dilate(canny, None, iterations=1)

cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] 
sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[1] * image.shape[1])

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
    
    # Resize ROI to 128x128
    roi = cv2.resize(roi, (128, 128))

    # Mark them on the image if you want
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),3)

    # Save your contours or characters
    cv2.imwrite("C:/input/character segmentation/e" + str(i) + ".png", roi)
    
    i = i + 1
    
cv2.imwrite("C:\input\malayalam\e1 character.jpg",orig)
plt.figure(figsize=(15,15))
plt.title("Character Segmetation")
plt.imshow(orig,'gray')
cnts.clear()


# In[3]:


import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = 'C:/input/models/model2/modelfull1-4.h5'
model = tf.keras.models.load_model(model_path)

# Load the label mapping from the training data generator
training_data_dir = 'C:/input/train'
train_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(training_data_dir, target_size=(128, 128), color_mode='grayscale', batch_size=32, class_mode='categorical')
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

# Predict images in a folder
image_folder_path = 'C:/input/character segmentation/'
predicted_labels = []
for filename in os.listdir(image_folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load and preprocess the image
        image_path = os.path.join(image_folder_path, filename)
        img = image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make a prediction
        prediction = model.predict(img_array,verbose=0)
        
        # Get the predicted label
        predicted_label = np.argmax(prediction, axis=1)[0]
        
        # Store the predicted label
        predicted_labels.append(predicted_label)

# Print the predicted labels and their corresponding class names in a single line
predicted_classes = [label_map[label] for label in predicted_labels]
print('Predicted classes:', ''.join(predicted_classes))
with open('C:/input/validation/predicted_classes.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(predicted_classes))


# In[4]



