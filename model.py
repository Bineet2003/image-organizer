import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set the path to your dataset
dataset_path = r'C:\Users\patel\OneDrive\Pictures'

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2  # 20% for validation
)

# Load training and validation data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # VGG16 input size
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'
)

validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # VGG16 input size
    batch_size=32,
    class_mode='categorical', 
    subset='validation'
)

# Load the pre-trained VGG16 model + higher level layers
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Create the complete model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(len(train_data.class_indices), activation='softmax')  # Number of categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10,  # You can adjust the number of epochs
    steps_per_epoch=len(train_data),
    validation_steps=len(validation_data)
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(validation_data)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Save the model
model.save('photo_organizer_model.h5')

# Function to predict the category of a new image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class


# Plotting training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
