
import os
import cv2
import numpy as np
import random
from sklearn.metrics import accuracy_score

# Define the folder path containing the new images
folder_path = "C:/Users/maram/OneDrive/Desktop/tajrba/"

# List to store the full paths of all images in the folder
installed_images_paths = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Construct the full path and add it to the list
        image_path = os.path.join(folder_path, filename)
        installed_images_paths.append(image_path)

# Shuffle the list of image paths randomly
random.shuffle(installed_images_paths)

# Load and preprocess the images
installed_images = []
image_size = (100, 100)
for image_path in installed_images_paths:
    image = cv2.imread(image_path)
    if image is not None:  # Check if the image was loaded successfully
        resized_image = cv2.resize(image, image_size)
        normalized_image = resized_image / 255.0  # Normalization
        installed_images.append(normalized_image)
    else:
        print(f"Skipping {image_path} due to read error")

if len(installed_images) == 0:
    print("No valid images found in the specified folder.")
else:
    # Convert to numpy array
    installed_images = np.array(installed_images)

    # Make predictions on the installed images
    predictions = model_VGG.predict(installed_images)  # Utilisez votre modèle ici
    # Get the predicted classes for each image
    predicted_classes = np.argmax(predictions, axis=1)
    # List to store predicted labels for each image
    predicted_labels = [classes[pred_class] for pred_class in predicted_classes]

    # Display images with their predictions
    print("\nImages with their predictions:")
    for idx, image_path in enumerate(installed_images_paths):
        prediction = predicted_labels[idx]
        print(f"Image {idx + 1}: {image_path}, Prediction: {prediction}")
   # Extract file names from true labels (full file paths)
    true_file_names = [os.path.basename(image_path).split('.')[0] for image_path in installed_images_paths]
    # Create a list of tuples containing the true file names and their corresponding predicted labels
    true_predicted_pairs = list(zip(true_file_names, predicted_labels))

    # Calculate accuracy
    true_labels = [filename.split('_')[0] for filename in true_file_names]
    predicted_labels = [label.split('_')[0] for label in predicted_labels]
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)



