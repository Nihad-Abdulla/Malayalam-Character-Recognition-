#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input shape of the images and the number of classes
input_shape = (128, 128, 1)
num_classes = 53   # English alphabet has 26 letters

# Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the training and testing data using ImageDataGenerator
train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_data.flow_from_directory('C:/input/train', target_size=input_shape[:2],
                                                  color_mode='grayscale', batch_size=32, shuffle=True,
                                                  class_mode='categorical', subset='training')
val_generator = train_data.flow_from_directory('C:/input/test', target_size=input_shape[:2],
                                                color_mode='grayscale', batch_size=32, shuffle=True,
                                                class_mode='categorical')
print("Training started...")
# Train the model
history = model.fit(train_generator, epochs=50, validation_data=val_generator)
print("Training completed.")

# Save the trained model
model.save('C:/input/models/model2/modelfull1-3.h5')

# Print the accuracy and loss of the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print("Validation accuracy:", val_acc)
print("Validation loss:", val_loss)


# In[27]:


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


# In[20]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input shape of the images and the number of classes
input_shape = (128, 128, 1)
num_classes = 53   # English alphabet has 26 letters

# Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the training and testing data using ImageDataGenerator
train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_data.flow_from_directory('C:/input/train', target_size=input_shape[:2],
                                                  color_mode='grayscale', batch_size=32, shuffle=True,
                                                  class_mode='categorical', subset='training')
val_generator = train_data.flow_from_directory('C:/input/test', target_size=input_shape[:2],
                                                color_mode='grayscale', batch_size=32, shuffle=True,
                                                class_mode='categorical')
print("Training started...")
# Train the model
history = model.fit(train_generator, epochs=30, validation_data=val_generator)
print("Training completed.")

# Save the trained model
model.save('C:/input/models/model2/modelfull1-4.h5')

# Print the accuracy and loss of the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print("Validation accuracy:", val_acc)
print("Validation loss:", val_loss)


# In[ ]:




