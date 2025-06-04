# -Image-Classification-using-Neural-Networks
In my image classification project, I used deep learning to enhance a model designed for classifying images within the CIFAR-10 dataset. It contains images of dogs, cats, aeroplane, frogs, deer and some more objects. By leveraging TensorFlow, I focused on refining the model’s architecture to boost accuracy while also speeding up the training process. This involved exploring different neural network setups, applying data augmentation, and fine-tuning hyperparameters—all aimed at reducing the number of epochs without sacrificing performance.
One of the key challenges I tackled was balancing the need for speed with the demand for precision, ensuring that the model could quickly and accurately classify images across various categories. I integrated a Flask-based backend, which allowed for easy deployment and real-time use, making the model applicable to real-world situations where fast and reliable image classification is crucial.
Throughout the project, I conducted rigorous testing and validation, continually refining the model to enhance its robustness. This experience not only deepened my understanding of deep learning and TensorFlow but also highlighted the importance of practical deployment strategies in machine learning.

---

## CNN Explanation 

🧠 What is a CNN?
A Convolutional Neural Network (CNN) is a specialized type of neural network primarily used for image data. It is particularly good at learning spatial hierarchies and patterns (like edges, textures, and shapes) directly from raw image pixels.

🧱 Basic CNN Architecture (Like in Your Project)
Your CNN model structure:

scss
Copy
Edit
Input (32x32x3 image)
↓
Conv2D (32 filters) + ReLU
↓
MaxPooling2D
↓
Conv2D (64 filters) + ReLU
↓
MaxPooling2D
↓
Flatten
↓
Dense (64) + ReLU
↓
Dense (10) + Softmax (for 10 classes)
Let’s break this down step-by-step:

1️⃣ Input Layer
Accepts 32x32 RGB images (shape: (32, 32, 3))

This comes from the CIFAR-10 dataset

2️⃣ Convolutional Layers (Conv2D)
🔹 What it does:
Applies filters (kernels) over the input image to detect local features like edges, corners, etc.

Each filter “slides” across the image and computes dot products to form a feature map.

🔹 Your config:
Conv2D(32, (3, 3)): 32 filters of size 3x3

Conv2D(64, (3, 3)): 64 filters in the second layer

🔹 Activation: ReLU
Applies the ReLU (Rectified Linear Unit) activation:
f(x) = max(0, x)

Removes negative values → adds non-linearity

3️⃣ Pooling Layers (MaxPooling2D)
🔹 What it does:
Downsamples the feature maps by selecting the maximum value in a small region (usually 2x2)

Reduces:

Dimensionality (fewer computations)

Overfitting (by making the network more robust)

🔹 Your config:
MaxPooling2D((2, 2)): Reduces each feature map by half

4️⃣ Flatten Layer
Converts the 2D output from the final pooling layer into a 1D vector

Example: From shape (8, 8, 64) → vector of 4096 elements

5️⃣ Dense Layers (Fully Connected Layers)
🔹 Dense(64) + ReLU:
Fully connected layer with 64 neurons

Learns complex combinations of features from previous layers

🔹 Dense(10) + Softmax:
Final output layer with 10 neurons (one per CIFAR-10 class)

Softmax activation gives a probability distribution over the 10 classes

🔁 Forward Pass Summary
Image is passed through convolution and pooling layers

Intermediate features (edges → shapes → objects) are extracted

Flattened and passed through dense layers for classification

🧪 Training Overview
Loss Function: sparse_categorical_crossentropy

Used for multi-class classification when labels are integers (0–9)

Optimizer: Adam

Efficient, adaptive gradient descent algorithm

Metrics: accuracy

🎯 Why CNNs Work Well for Images
Parameter sharing: Convolution filters are reused across the image, reducing the number of trainable parameters.

Local connectivity: Each neuron in the conv layer is connected to only a small region of the input (receptive field), allowing the network to learn spatial hierarchies.

Translation invariance: Learned filters can detect features anywhere in the image.

🖼 Example of What CNN Learns
Layer	Learns
Conv1	Simple edges (horizontal, vertical)
Conv2	Corners, curves
Conv3+Dense	High-level object patterns


---

🔍 Project Explanation 

📁 1. Dataset: CIFAR-10
CIFAR-10 is a publicly available dataset containing 60,000 32x32 color images in 10 classes (6,000 images per class).

There are 50,000 training images and 10,000 test images.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

🔗 CIFAR-10 Dataset on TensorFlow Docs

🧠 2. Model: CNN (Convolutional Neural Network)
You built a simple yet effective CNN using tensorflow.keras:

Conv2D: 32 filters (3x3) with ReLU activation

MaxPooling2D: Downsamples the output

Conv2D: 64 filters (3x3)

MaxPooling2D

Flatten: Converts 2D feature maps to 1D

Dense: 64 neurons with ReLU

Dense: 10 neurons with Softmax for classification output

Compiled with:

Optimizer: adam

Loss function: sparse_categorical_crossentropy

Metric: accuracy

🏋️ 3. Training
The model is trained for 8 epochs using the CIFAR-10 dataset.

Input images are normalized by dividing pixel values by 255.

The trained model is saved as cnn_model.h5.

🌐 4. Web Interface (Flask App)
You used Flask to build a web server with two routes:

/: Main route that serves the image upload form (imgclass.html)

/result: Displays prediction (result.html)

📤 5. Uploading and Predicting
When a user uploads an image:

It’s resized to 32x32

Converted to a NumPy array and normalized

Fed into the model for prediction

The predicted class name is returned (e.g., dog, ship)

🖥️ 6. Frontend (HTML + CSS)
Two HTML templates are used:

imgclass.html: File upload interface

result.html: Displays predicted class and class index

Styled with modern dark theme using CSS.

📁 7. File Management
Uploaded files are saved in a local uploads/ folder (created if it doesn't exist).

Prediction is made using the stored model without retraining.

✅ 8. Output
The user sees a result page with:

Predicted class name

Class index (optional)

Button to upload a new image

📈 9. Evaluation
The model is evaluated using the test set:

python
Copy
Edit
cnn.evaluate(x_test, y_test)
📌 10. Deployment (Optional Future Steps)
Can be deployed using:

Render, Heroku, PythonAnywhere

Docker container for consistent environments

---

# 🧠 CIFAR-10 Image Classification Web App

A deep learning web application built with **TensorFlow**, **Keras**, and **Flask** that classifies images into one of the 10 CIFAR-10 categories: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

---

## 📌 Table of Contents

* [📌 Table of Contents](#-table-of-contents)
* [🚀 Overview](#-overview)
* [🧰 Technologies Used](#-technologies-used)
* [📂 Project Structure](#-project-structure)
* [📦 Setup Instructions](#-setup-instructions)
* [🧪 Model Training](#-model-training)
* [🌐 Web Interface](#-web-interface)
* [🖼️ Sample Results](#-sample-results)
* [📸 Screenshots](#-screenshots)
* [💡 Future Improvements](#-future-improvements)
* [📃 License](#-license)

---

## 🚀 Overview

This project demonstrates how to:

* Train a CNN model on the CIFAR-10 dataset using TensorFlow.
* Save and load the trained model.
* Develop a Flask web interface for uploading and classifying new images.
* Display classification results on a styled HTML result page.

---

## 🧰 Technologies Used

| Component     | Tech Stack                         |
| ------------- | ---------------------------------- |
| Language      | Python 3.x                         |
| Libraries     | TensorFlow, NumPy, Matplotlib      |
| Web Framework | Flask                              |
| Frontend      | HTML, CSS (custom styling), Jinja2 |
| Dataset       | CIFAR-10 (via Keras API)           |

---

## 📂 Project Structure

```
cifar10-flask/
│
├── templates/
│   ├── imgclass.html          # Upload page
│   └── result.html            # Classification result page
│
├── uploads/                   # Uploaded image files (auto-created)
├── cnn_model.h5               # Trained model
├── app.py                     # Flask application
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## 📦 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/cifar10-flask.git
cd cifar10-flask
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Flask app**

```bash
python app.py
```

5. Open your browser and visit: `http://127.0.0.1:5000`

---

## 🧪 Model Training

The CNN model is trained on the CIFAR-10 dataset with the following architecture:

* Conv2D (32 filters) → ReLU → MaxPooling
* Conv2D (64 filters) → ReLU → MaxPooling
* Flatten → Dense(64) → ReLU
* Output Layer: Dense(10, softmax)

Training is done for 8 epochs using the Adam optimizer and sparse categorical cross-entropy loss.

You can retrain or modify the model by editing the `app.py` script.

---

## 🌐 Web Interface

* **Upload Page** (`imgclass.html`): Allows user to upload an image.
* **Result Page** (`result.html`): Displays predicted class name and class index.

---

## 🖼️ Sample Results

| Uploaded Image | Predicted Class |
| -------------- | --------------- |
| `cat.jpg`      | `cat`           |
| `car.jpg`      | `automobile`    |
| `frog.png`     | `frog`          |

---

## 💡 Future Improvements

* Add drag-and-drop image upload.
* Support batch image predictions.
* Add top-3 prediction probabilities.
* Improve CNN architecture using techniques like dropout or batch normalization.
* Deploy on a cloud platform (Heroku, Render, or AWS).

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

## Requirements.txt

Flask
tensorflow
numpy
matplotlib
Pillow


---

> If you found this project helpful, please ⭐ the repo and share it!

---



