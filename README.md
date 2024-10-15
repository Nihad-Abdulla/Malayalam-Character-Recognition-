Project Overview

This project aims to develop a machine learning model using Convolutional Neural Networks (CNN) to recognize and digitize handwritten Malayalam characters. Malayalam, a Dravidian language with 53 unique characters, including vowels and consonants, presents challenges for character recognition due to the diverse styles of handwriting.

The project draws inspiration from Optical Character Recognition (OCR), commonly used for printed text. However, traditional OCR methods struggle with handwritten text, especially for regional scripts like Malayalam. To address this, the project uses CNNs, which excel in image recognition by identifying patterns and features. The CNN model is trained on a dataset of handwritten Malayalam characters, using techniques like image pre-processing, grayscale conversion, and noise reduction to enhance recognition accuracy and classify characters based on their visual features.


Project Goal


The goal is to create a tool for digitizing handwritten Malayalam text, simplifying document management in sectors like customer service and education that still rely on paper forms. This solution will make archiving, searching, and handling handwritten documents faster and more efficient, bridging the gap in OCR systems for regional languages.


Steps

Pre-Processing

In this project, pre-processing is essential for preparing handwritten Malayalam character images for recognition by the CNN model. The images are first converted to grayscale to simplify processing, followed by Gaussian Blur to reduce noise. Otsu's thresholding is then applied to convert the image into a binary format, distinguishing the characters from the background. To further refine the image, morphological operations like closing and opening are used to remove small artifacts and enhance the clarity of the character shapes. These steps ensure that the image is clean and ready for the CNN to accurately recognize the characters.

Character Segmentation

After pre-processing, the next step is Character Segmentation, which isolates individual characters from the image for recognition. Edge detection is performed using Canny Edge Detection to highlight the boundaries of the characters. The edges are then dilated to enhance the character boundaries, followed by contour detection to identify individual character regions. Each contour represents a potential character, and contours with very small areas are filtered out to ignore noise. The identified characters are extracted using bounding boxes, resized to a standard size of 128x128 pixels, and saved as separate images. This ensures that each segmented character is ready for further processing and classification by the CNN model.

Model Design , Training 


The model design involves building a Convolutional Neural Network (CNN) to recognize and classify the segmented Malayalam characters. The CNN architecture consists of multiple convolutional layers that extract features from the input images, followed by pooling layers to reduce dimensionality. The model includes four convolutional layers with increasing filters (32, 64, 128, and 256), each followed by max-pooling layers. After feature extraction, the output is flattened and passed through two fully connected dense layers with ReLU activation, and the final layer uses softmax activation to classify the characters into one of the 53 classes.

The model is compiled using the Adam optimizer and categorical cross-entropy loss function, suitable for multi-class classification problems. The training and validation data are pre-processed and loaded using ImageDataGenerator with rescaling. The model is trained for 30 epochs, with real-time validation data used to monitor performance. After training, the model is saved for future use, and the validation accuracy and loss are computed to evaluate the model's effectiveness.

Evaluation

The evaluation code effectively loads a trained TensorFlow model and processes images for character recognition from a specified folder. It maps predicted labels to their corresponding class names and calculates confidence scores for each prediction. By utilizing batch processing, the code improves efficiency, enabling quicker predictions across multiple images. The results are printed with their respective confidence levels and saved to a text file, providing a clear overview of the model's performance in identifying characters. This streamlined approach enhances both accuracy and usability in evaluating the model's effectiveness.
