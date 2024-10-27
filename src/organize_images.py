import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import shutil
import sys

model = tf.keras.models.load_model('/app/image_classifier.h5')
CLASS_NAMES = ['Notes & Documents', 'Selfie', 'Family & Friends', 'Memes & Non-personal Photos']

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    return CLASS_NAMES[np.argmax(predictions)]

def organize_images(folder_path):
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(folder_path, class_name), exist_ok=True)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        predicted_class = predict_image(file_path)
        shutil.move(file_path, os.path.join(folder_path, predicted_class, filename))
        print(f'Moved {filename} to {predicted_class}/')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_images.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1]
    organize_images(folder_path)
