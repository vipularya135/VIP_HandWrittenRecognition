import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import string

# Load the trained model
model = keras.models.load_model('character_recognition_model.h5')

# Path to the input image
image_path = r'C:\Users\DELL\Desktop\char_recog\sentence_image.jpg'

# Load and preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to (256, 256) for VGG19 input
    resized_image = cv2.resize(gray_image, (256, 256))
    # Convert grayscale to RGB by stacking the channels
    rgb_image = cv2.merge([resized_image] * 3)  # Stack the grayscale image to create 3 channels
    # Normalize pixel values
    normalized_image = rgb_image.astype('float32') / 255.0
    # Add batch dimension
    return np.expand_dims(normalized_image, axis=0)  # (1, 256, 256, 3)

# Predict the characters in the image
def predict_sentence(image_path):
    preprocessed_image = preprocess_image(image_path)
    
    # Predict characters using the model
    predictions = model.predict(preprocessed_image)
    predicted_classes = np.argmax(predictions, axis=-1)  # Get the predicted classes

    return predicted_classes  # Returns a single value or array depending on model output

# Mapping from class indices to actual characters (update according to your setup)
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits  # Assuming you have A-Z, a-z, 0-9
# Add space or other characters if necessary
# characters += ' '  # If your model predicts space

# Main code to predict and print the recognized sentence
predicted_classes = predict_sentence(image_path)

# Check if the output is a single class or multiple classes
if isinstance(predicted_classes, np.ndarray):
    recognized_sentence = ''.join([characters[i] for i in predicted_classes])
else:
    recognized_sentence = characters[predicted_classes]  # If single class prediction

print(f"Recognized Sentence: {recognized_sentence}")

# Optionally, display the original image
original_image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis('off')
plt.show()
