import os
from utils import create_directories

def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            images.append(os.path.join(image_folder, filename))
    return images

def organize_images(image_folder):
    categories = ['Notes & Documents', 'Selfie', 'Family & Friends', 'Memes & Non-personal Photos']
    create_directories(categories)
