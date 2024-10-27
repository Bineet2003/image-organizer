from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'trained_model/image_classifier.h5'
CLASS_NAMES = ['Notes & Documents', 'Selfie', 'Family & Friends', 'Memes & Non-personal Photos']
model = tf.keras.models.load_model(MODEL_PATH)

def sort_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img = load_img(os.path.join(input_folder, filename), target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_name = CLASS_NAMES[np.argmax(prediction)]

            output_folder = os.path.join(input_folder, class_name)
            os.makedirs(output_folder, exist_ok=True)
            shutil.move(os.path.join(input_folder, filename), os.path.join(output_folder, filename))
            print(f'Moved {filename} to {class_name}')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        sort_images(folder_path)
        return f'Images sorted in {folder_path}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
