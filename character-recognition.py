import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import string
import seaborn as sns
import os
import cv2

# Load and preprocess dataset
train_ds = image_dataset_from_directory(
    directory='handwritten-english-characters-and-digits/combined_folder/train',
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    shuffle=True
)

test_ds = image_dataset_from_directory(
    directory='handwritten-english-characters-and-digits/combined_folder/test',
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    shuffle=True
)

# Load pre-trained VGG19 for feature extraction
conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
conv_base.trainable = False

# Build and compile the model
model = Sequential([
    Input(shape=(256, 256, 3)),
    conv_base,
    Flatten(),
    Dense(224, activation='relu'),
    Dropout(0.1),
    Dense(416, activation='sigmoid'),
    Dropout(0.1),
    Dense(62, activation='softmax')  # 62 classes for digits, uppercase, and lowercase
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=15, validation_data=test_ds)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the trained model
model_save_path = 'character_recognition_model.h5'
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# Load the saved model
model = keras.models.load_model(model_save_path)

# Define character classes (0-9 for digits, A-Z for uppercase, a-z for lowercase)
digits = [str(i) for i in range(10)]
uppercase = [i for i in string.ascii_uppercase]
lowercase = [i for i in string.ascii_lowercase]
class_names = digits + uppercase + lowercase

# Function to preprocess the image and extract characters
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get a binary image (black text, white background)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create bounding boxes around characters and store the ROI (Region of Interest)
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the character ROI and resize it to the input size of the model (256x256)
        char_roi = gray[y:y+h, x:x+w]
        resized_char = cv2.resize(char_roi, (256, 256))
        char_images.append(resized_char)

    return char_images

# Function to predict each character from its image
def predict_character(char_img):
    char_img = np.expand_dims(char_img, axis=-1)  # Add the channel dimension (grayscale)
    char_img = np.repeat(char_img, 3, axis=-1)    # Convert grayscale to 3 channels (as the model expects RGB)
    char_img = np.expand_dims(char_img, axis=0)   # Add the batch dimension
    char_img = char_img / 255.0                   # Normalize the image

    # Predict using the loaded model
    predictions = model.predict(char_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class

# Function to recognize a sentence from a handwritten image
def recognize_sentence(image_path):
    char_images = preprocess_image(image_path)
    sentence = ""

    # Sort the characters based on their x-coordinate to maintain the correct sequence
    sorted_chars = sorted(char_images, key=lambda img: cv2.boundingRect(img)[0])

    for char_img in sorted_chars:
        predicted_char = predict_character(char_img)
        sentence += predicted_char
    
    return sentence

# Example usage: recognizing a sentence from a handwritten image
image_path = 'sentence_image.jpg'  # Specify your image path here
recognized_sentence = recognize_sentence(image_path)
print(f"Recognized Sentence: {recognized_sentence}")
