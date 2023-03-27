#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Set the batch size, number of epochs, and target image size
batch_size = 5
num_epochs = 10
target_size = (224, 224)

# Set the directory paths for training, validation, and testing
train_dir = 'C:/input/training'
valid_dir = 'C:/input/validation'
test_dir = 'C:/input/testing'

# Set the label for the character
char_label = 'ന'

# Create the data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary')

print("Number of training samples:", train_generator.n)
print("Number of validation samples:", valid_generator.n)
print("Number of testing samples:", test_generator.n)


# Build the CNN model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model using the data generator for training and validate on the validation set
print('Training the model...')
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    verbose=1)

# Evaluate the model on the testing set
print('Evaluating the model...')
test_loss, test_acc = model.evaluate(
    test_generator, 
    steps=len(test_generator),
    verbose=1)

print('Test accuracy:', test_acc)

# Save the model with the character label as the filename
model.save('C:/input/models/model1/model.h5')

print('Model for character ' + char_label + ' saved to C:/input/models/model1.')


# In[2]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Set the batch size, number of epochs, and target image size
batch_size = 5
num_epochs = 10
target_size = (224, 224)

# Set the directory paths for training, validation, and testing
train_dir = 'C:/input/training'
valid_dir = 'C:/input/validation'
test_dir = 'C:/input/testing'

# Set the label for the character
char_label = 'ന'

# Create the data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

print("Number of training samples:", train_generator.n)
print("Number of validation samples:", valid_generator.n)
print("Number of testing samples:", test_generator.n)


# Build the CNN model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model using the data generator for training and validate on the validation set
print('Training the model...')
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    verbose=1)

# Evaluate the model on the testing set
print('Evaluating the model...')
test_loss, test_acc = model.evaluate(
    test_generator, 
    steps=len(test_generator),
    verbose=1)

print('Test accuracy:', test_acc)

# Save the model with the character label as the filename
model.save('C:/input/models/model1/model.h5')

print('Model for character ' + char_label + ' saved to C:/input/models/model1.')



# In[72]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Set the batch size, number of epochs, and target image size
batch_size = 5
num_epochs = 10
target_size = (224, 224)

# Set the directory paths for training, validation, and testing
train_dir = 'C:/input/training'
valid_dir = 'C:/input/validation'
test_dir = 'C:/input/testing'

# Set the label for the character
char_label = 'ന'

# Create the data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary')

print("Number of training samples:", train_generator.n)
print("Number of validation samples:", valid_generator.n)
print("Number of testing samples:", test_generator.n)


# Build the CNN model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()


# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model using the data generator for training and validate on the validation set
print('Training the model...')
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    verbose=1)

# Evaluate the model on the testing set
print('Evaluating the model...')
test_loss, test_acc = model.evaluate(
    test_generator, 
    steps=len(test_generator),
    verbose=1)

print('Test accuracy:', test_acc)

# Get the predicted and true labels for the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Print the classification report and confusion matrix
print('Classification Report')
print(classification_report(y_true, y_pred))

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))

# Save the model with the character label as the filename
model.save('C:/input/models/model2/model.h5')

import matplotlib.pyplot as plt

# Plot the training and validation loss over the number of epochs
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[10]:


from PIL import Image

# Open the image and get its mode
img = Image.open('C:/input/character segmentation/e1char3.png')
mode = img.mode

# Print the mode to determine the number of channels
print(mode)


# In[ ]:





# In[64]:


# Load the saved model
saved_model = load_model('C:/input/models/model2/model.h5')

# Load the image and preprocess it
img_path = 'C:\input\character segmentation\IMG187.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, target_size)
img = np.reshape(img, (1, 224, 224, 3))

# Make a prediction
prediction = saved_model.predict(img)

# Extract the label from the saved model
label_map = train_generator.class_indices
label_map = dict((v,k) for k,v in label_map.items())
label = label_map[int(prediction[0][0])]

print('The predicted label is:', label)


# In[76]:


import os
import numpy as np
import cv2
from keras.models import load_model

# Set the target image size
target_size = (224, 224)

# Load the saved model
saved_model = load_model('C:/input/models/model2/model.h5')

# Define the character label
char_label = saved_model.output_names[0]

# Load the test image
img_path = 'C:\input\character segmentation\IMG187.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, target_size)
img = np.reshape(img, (1, 224, 224, 3))

# Make a prediction on the image using the saved model
prediction = saved_model.predict(img)

# Get the label of the prediction
if prediction[0][0] > 0.5:
    label = char_label
else:
    label = 'Not ' + char_label

# Print the label of the prediction
print('Prediction:', label)


# In[ ]:





# In[ ]:




