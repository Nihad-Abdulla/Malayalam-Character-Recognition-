


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
# Set the batch size, number of epochs, and target image size
# Set the batch size, number of epochs, and target image size
batch_size = 5
num_epochs = 10
target_size = (128, 128)

# Set the directory paths for training, validation, and testing
train_dir = 'C:/input/training'
valid_dir = 'C:/input/validation'
test_dir = 'C:/input/testing'

# Set the label for the character
char_label = 'na' #TODO 'ന' na (then while printing add a switch_case_)

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

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
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
model.save('C:/input/models/model1/model.h5')

print('Model for character ' + char_label + ' saved to C:/input/models/model1.')

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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model_path = 'C:/input/models/model2/model.h5'
model = tf.keras.models.load_model(model_path)

# Load the image and preprocess it for prediction
image_path = 'C:/input/character segmentation/IMG149.jpg'
img = image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make a prediction
prediction = model.predict(img_array)

# Get the predicted label
predicted_label = np.argmax(prediction, axis=1)[0]

# Print the predicted label
print(f'The predicted label is {predicted_label}.')


# In[26]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input shape of the images and the number of classes
input_shape = (128, 128, 1)
num_classes = 3  # English alphabet has 26 letters

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
train_generator = train_data.flow_from_directory('C:/input/training', target_size=input_shape[:2],
                                                  color_mode='grayscale', batch_size=32, shuffle=True,
                                                  class_mode='categorical', subset='training')
val_generator = train_data.flow_from_directory('C:/input/testing', target_size=input_shape[:2],
                                                color_mode='grayscale', batch_size=32, shuffle=True,
                                                class_mode='categorical')
print("Training started...")
# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)
print("Training completed.")

# Save the trained model
model.save('C:/input/models/model2/model.h5')

# Print the accuracy and loss of the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print("Validation accuracy:", val_acc)
print("Validation loss:", val_loss)


# In[2]:


from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = 'C:/input/models/model2/model.h5'
model = tf.keras.models.load_model(model_path)

# Load the image and preprocess it for prediction
image_path = 'C:/input/character segmentation/IMG244.jpg'
img = image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make a prediction
prediction = model.predict(img_array)

# Get the predicted label
predicted_label = np.argmax(prediction, axis=1)[0]

# Get the label mapping from the training data generator
training_data_dir = 'C:/input/training'
train_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(training_data_dir, target_size=(128, 128), color_mode='grayscale', batch_size=32, class_mode='categorical')
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

# Print the predicted label and its corresponding class name
print(f'The predicted label is {predicted_label}.')
print(f'The predicted class is {label_map[predicted_label]}.')


# In[ ]:




