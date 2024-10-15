#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

