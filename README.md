# WhatsApp Image Classification using Transfer Learning

This project implements a **multi-class image classification** model to categorize WhatsApp images into the following classes:  
- **Notes & Documents**  
- **Selfie**  
- **Family & Friends**  
- **Memes & Non-personal Photos**  

We leverage **InceptionV3**, a pre-trained convolutional neural network (CNN), to build a transfer learning-based solution. The model extracts high-level features from images and fine-tunes additional dense layers to improve performance.

---

## Key Features  

- **Transfer Learning:**  
  Utilizes **InceptionV3** pre-trained on ImageNet to save training time and achieve higher accuracy.  

- **Data Augmentation:**  
  Augmentation techniques, including rotation, zoom, shear, and horizontal flips, are applied to prevent overfitting and improve generalization.

- **Batch Normalization & Dropout:**  
  Regularization techniques ensure the model remains stable during training and avoids overfitting.  

- **Evaluation Metrics:**  
  **Accuracy**, **loss graphs**, and a **confusion matrix** are used to measure model performance on both training and validation datasets.

---

## Dependencies  
The following libraries are required to run this project:  
- **TensorFlow** for building and training the model  
- **Pandas & NumPy** for data handling  
- **Matplotlib & Seaborn** for plotting results  
- **scikit-learn** for generating the classification report  
