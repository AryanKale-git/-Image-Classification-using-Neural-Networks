import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing import image
import os

# Load CIFAR-10 dataset and preprocess it
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Function to plot sample images
def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

# Define CNN model
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train, y_train, epochs=8)
cnn.evaluate(x_test, y_test)

# Save the model
model_path = 'cnn_model.h5'
cnn.save(model_path)

# Initialize Flask app
app = Flask(__name__)

def prepare_image(file):
    img = image.load_img(file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            img = prepare_image(filepath)
            model = models.load_model(model_path)
            prediction = model.predict(img)
            predicted_class = classes[np.argmax(prediction)]
            return render_template('result.html', predicted_class=predicted_class)
    return render_template('imgclass.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, port=5000)

