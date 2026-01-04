# TensorFlow 
## Index
- [Module 1: Basics](#module-1)
- [Module 2: Classification](#module-2)
- [Module 3: Convolutional Neural Networks](#module-3)
- [Module 4: Reinforcement Learning](#module-4)
- [Projects](#projects)
- [Models](#models)
- [Setup](#setup)


This repository is my personal TensorFlow learning sandbox.
Each module represents a concrete step forward - from understanding tensors and preprocessing data to building CNNs, NLP models, and reinforcement learning agents.

This is not a polished library or a tutorial series.
It’s a transparent, hands-on log of learning TensorFlow by building real models, saving checkpoints, and measuring results.


---
<p align="left">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=3000&pause=500&color=00FF80&background=765898&center=false&vCenter=false&width=440&lines=Basics;Classification;Convolution+Neural+Network+(CNN);Reinforcement-Learning;Project:+Sentiment+Analysis;Project:+Fuel+Efficiency;Project:+Fashion+MNIST" alt="Typing SVG" />
</p>

---
## Module 1
### [Basics](https://github.com/aypy01/tensorflow/blob/main/module-1.ipynb)

**Key learnings:**
- Difference between `tf.Tensor` (immutable) and `tf.Variable` (mutable, required for trainable weights)
- Tensor slicing behavior vs reassignment limitations
- Practical use of `reshape` when adapting data to model input requirements
- Data quality issues belong to preprocessing, not model or architecture choice

---

## Module 2
### [Classification](https://github.com/aypy01/tensorflow/blob/main/module-2.ipynb)

**Key learnings:**
- Built end-to-end classification pipelines for:
  - **Titanic** (binary classification)
  - **Iris** (multiclass classification)
- Proper train/test splitting (`x_train`, `y_train`, `x_test`, `y_test`)
- Handling missing values using `fillna`
- Feature preprocessing:
  - Dropped non-informative features (e.g., `fare`)
  - Scaled numerical features with `StandardScaler`
  - One-hot encoded categorical features
- Model design:
  - Sigmoid output + binary crossentropy for binary classification
  - Softmax output + sparse categorical crossentropy for multiclass classification
- Used TensorFlow’s `Normalization` layer instead of external scalers for Iris
- Saved trained models as `.keras` files
- Achieved ~81% accuracy (Titanic) and ~70% accuracy (Iris)

---

## Module 3
### [Convolutional Neural Networks (CNN)](https://github.com/aypy01/tensorflow/blob/main/module-3.ipynb)

**Key learnings:**
- Built CNNs from scratch for image classification using CIFAR-10
- Normalized image inputs to the 0–1 range
- Stacked `Conv2D`, `MaxPooling`, and `Dense` layers for feature extraction and classification
- Achieved ~72% accuracy on CIFAR-10 with a baseline CNN
- Applied image augmentation (zoom, shift, rotation) to improve generalization
- Observed trade-offs between augmentation and raw accuracy (~61%)
- Implemented transfer learning using MobileNetV2 (`weights="imagenet"`, `include_top=False`)
- Preprocessed inputs to the [-1, 1] range as required by MobileNetV2
- Used `GlobalAveragePooling` and a custom classification head
- Achieved ~94% accuracy on Dogs vs Cats, highlighting the effectiveness of transfer learning

---

## Module 4
### [Reinforcement Learning](https://github.com/aypy01/tensorflow/blob/main/module-4.ipynb)

**Key learnings:**
- Implemented Q-learning on the `FrozenLake-v1` environment using Gymnasium
- Initialized and updated a Q-table using the Bellman equation
- Applied an epsilon-greedy strategy to balance exploration and exploitation
- Tuned hyperparameters: learning rate (α), discount factor (γ), exploration rate (ε)
- Implemented epsilon decay to shift from exploration to exploitation
- Evaluated agent performance by averaging rewards across episodes
- Achieved ~72% success rate on FrozenLake

---

## Projects:
### [Sentiments](https://github.com/aypy01/tensorflow/blob/main/sentiments.ipynb)
>  The IMDB dataset was directory-based (`train/pos`, `train/neg`, `test/pos`, `test/neg`), with labels inferred from folder names and an unused `unsup` folder. I loaded it using `text_dataset_from_directory` with a validation split, seed, and batch size. Reviews were preprocessed using a custom standardization function (lowercasing, removing punctuation, stripping `<br />` tags) and a `TextVectorization` layer with `max_tokens=10000` and `sequence_length=250` to map words into integers. The model architecture included an Embedding layer, Dropout, GlobalAveragePooling1D, and a Dense output layer with sigmoid activation, compiled with Adam and binary crossentropy. After training with validation split, the model achieved \~81.9% accuracy on the test set and was saved as `sentiments.keras`.
### [Fuel Efficiency](https://github.com/aypy01/tensorflow/blob/main/fuel_efficiency.ipynb)
>  I prepared the fuel efficiency dataset by dropping irrelevant columns, handling missing values, and one-hot encoding categorical origin data. After splitting into train/test sets and separating features from the target (MPG), I normalized the numerical features using TensorFlow’s Normalization layer. A simple regression model was built with a Normalizer input and a single Dense output neuron, compiled with Adam and mean absolute error. The model achieved ~ 1.81 MAE on the test set, showing good baseline performance, and was saved as regression.keras.

### [Fashion MNIST](https://github.com/aypy01/tensorflow/blob/main/fashion-mnist-image-classifier.py)
>  I trained a CNN on the FashionMNIST dataset, normalizing 28×28 grayscale images to [0,1] and using Conv2D, MaxPooling, Dropout, and Dense layers with softmax for classification. Compiled with Adam and sparse categorical crossentropy, the model trained for 10 epochs with a validation split and achieved ~92.5% accuracy on the test set. Predictions correctly mapped review images to their classes, and the model was saved as fashion_mnist.keras.
## [Models](https://github.com/aypy01/tensorflow/tree/main/models)

> This folder contains the trained `.keras` models from different TensorFlow projects in this repository.  
They are saved checkpoints of my experiments - ready to be reloaded for evaluation, predictions, or fine-tuning.

> ### Model Index

| Model File              | Task / Dataset          | Metric Achieved     |
|--------------------------|-------------------------|---------------------|
| `titanic.keras`         | Binary Classification (Titanic survival) | ~81% Accuracy |
| `iris_species.keras`    | Multiclass Classification (Iris dataset) | ~70% Accuracy |
| `cifar10.keras`         | Image Classification (CIFAR-10) | ~72% Accuracy |
| `cifar_augmented.keras` | CIFAR-10 with Data Augmentation | ~61% Accuracy |
| `dogsvscat.keras`       | Transfer Learning (Dogs vs Cats, MobileNetV2) | ~94% Accuracy |
| `sentiments.keras` | Text Vectorization (IMDb reviews)  | 85.80 % Accuracy |
| `fuel_efficiency.keras` | Regression (Auto MPG dataset) | 1.81 MAE |
| `fashion_mnist.keras`   | Image Classification (Fashion MNIST) | ~92.5% Accuracy |

---
## Setup

Install dependencies:

```bash
pip install -r requirements.txt

```
---
## Usage

### To load and use any model:  

```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/fashion_mnist.keras")

# Evaluate or predict
loss, acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {acc:.2f}")

```

## Tech Stack
- TensorFlow / Keras
- Python 3

---

## Note


This repository documents learning through direct experimentation.
The focus is on building, breaking, debugging, and measuring - not on theoretical completeness or production readiness.

The repo is intentionally iterative and unfinished.
If you’re exploring TensorFlow, feel free to fork it, experiment, and adapt the ideas into your own projects.

---

## Author
 <p align="left">
  Created and maintained by &nbsp;
  <a href="https://github.com/aypy01" target="_blank">
  <img src="https://img.shields.io/badge/Aaditya_Yadav-aypy01-e6770b?style=flat-square&logo=github&logoColor=00FF80&labelColor=765898" alt="GitHub Badge"/>
</a>

</p>

<p>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=3000&pause=500&color=00FF80&background=765898&center=false&vCenter=false&width=440&lines=Break+Things+First%2C+Understand+Later;Built+to+Debug%2C+Not+Repeat;Learning+What+Actually+Sticks;Code.+Observe.+Refine." alt="Typing SVG" />
</p>

---

## License

This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT).



