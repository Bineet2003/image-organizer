from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import build_model

def load_data(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_data = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    val_data = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    return train_data, val_data

train_data, val_data = load_data('data/train', 'data/val')
model = build_model(num_classes=4)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)
model.save('src/image_classifier.h5')
