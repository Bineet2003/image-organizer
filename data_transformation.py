import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(299, 299))  # InceptionV3 expects 299x299 images
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images) / 255.0  # Normalize the images
