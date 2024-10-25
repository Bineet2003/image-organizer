import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Set paths for your training data and organized images
train_data_dir = r'C:\Users\patel\OneDrive\Pictures\Training Data'  # Directory containing training images
organized_path = r'C:\Users\patel\OneDrive\Pictures\Organized'  # Where to save organized images

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Flow training images in batches
train_data = data_gen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the VGG16 model without the top layer (classifier)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the new model
model = Sequential([
    base_model,  # Add the VGG16 base model
    Flatten(),   # Flatten the output from the base model
    Dense(128, activation='relu'),  # Add a fully connected layer
    Dense(len(train_data.class_indices), activation='softmax')  # Output layer for categories
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=50, steps_per_epoch=len(train_data))

# Save the trained model
model.save('photo_organizer_model.h5')
print("Model saved successfully!")

# Load the saved model
loaded_model = load_model('photo_organizer_model.h5')
print("Model loaded successfully!")

# Function to predict the category of a new image
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Loop through the images in the Saved Pictures directory
saved_pictures_path = r'C:\Users\patel\OneDrive\Pictures\Saved Pictures'

for filename in os.listdir(saved_pictures_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        img_path = os.path.join(saved_pictures_path, filename)
        
        # Predict the category for the image
        predicted_category_index = predict_image(img_path)

        # Get category name from class indices
        category_name = list(train_data.class_indices.keys())[predicted_category_index]

        # Create the category folder if it doesn't exist
        category_folder_path = os.path.join(organized_path, category_name)
        if not os.path.exists(category_folder_path):
            os.makedirs(category_folder_path)

        # Move the image to the category folder
        shutil.move(img_path, os.path.join(category_folder_path, filename))

print("Images have been organized successfully!")
