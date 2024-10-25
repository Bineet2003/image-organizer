import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('photo_organizer_model.h5')

# Set the path to the saved pictures directory
saved_pictures_path = r'C:\Users\patel\OneDrive\Pictures\Saved Pictures'
organized_path = r'C:\Users\patel\OneDrive\Pictures\Organized'

# Create the organized directory if it doesn't exist
if not os.path.exists(organized_path):
    os.makedirs(organized_path)

# Function to predict the category of a new image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the predicted class index

# Loop through the images in the Saved Pictures directory
for filename in os.listdir(saved_pictures_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Check for image file types
        img_path = os.path.join(saved_pictures_path, filename)
        
        # Predict the category for the image
        predicted_category_index = predict_image(img_path)

        # Get category name from class indices
        category_name = list(os.listdir(saved_pictures_path))[predicted_category_index]

        # Create the category folder if it doesn't exist
        category_folder_path = os.path.join(organized_path, category_name)
        if not os.path.exists(category_folder_path):
            os.makedirs(category_folder_path)

        # Move the image to the category folder
        shutil.move(img_path, os.path.join(category_folder_path, filename))

print("Images have been organized successfully!")
