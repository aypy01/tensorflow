# TensorFlow 
This repository is my personal sandbox for deep learning. Each project here represents a step forward from wrestling with data preprocessing and model architectures to debugging late night training runs. The modules and projects are not just exercises; they’re building blocks, each one helping me nderstand how raw data transforms into working intelligence. Think of this repo as a transparent log of progress: no polish, just the real work of learning TensorFlow by building with it.

---
<p align="left">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=3000&pause=500&color=00FF80&background=765898&center=false&vCenter=false&width=440&lines=Basics;Classification;Convolution+Neural+Network+(CNN);Reinforcement-Learning;Project:+Sentiment+Analysis;Project:+Fuel+Efficiency;Project:+Fashion+MNIST" alt="Typing SVG" />
</p>

---
## Module 1:
### [Basics](https://github.com/aypy01/tensorflow/blob/main/module-1.ipynb)
> Module 1 mainly circled around the basics of TensorFlow vs NumPy, the idea of tensors being immutable (like tuples) and tf.Variable being mutable (needed for weights that update), slicing feeling familiar yet tricky because reassignment only works with variables, and the bigger gap you noticed in your own understanding: you’ve practiced shapes but never really used reshape for a real purpose and now you know it matters only when data has to be bent into the exact form a model expects, not for its own sake. You also clarified that messy or unbalanced data isn’t a reason to “go deep learning”  that’s preprocessing work, not architecture choice.
## Module 2:
### [Classification](https://github.com/aypy01/tensorflow/blob/main/module-2.ipynb)
> Worked with two datasets from TensorFlow: Titanic (binary classification) and Iris (multiclass classification). For Titanic, I learned to properly split train/test sets (x_train, y_train, x_test, y_test) and handled missing values with fillna. I dropped the fare column (not useful for survival), scaled numerical features (age, etc.) with StandardScaler, and one-hot encoded categorical columns. After preprocessing, I built a neural network with input shape = number of features, hidden layers, and a single sigmoid output unit for binary classification. The model was compiled with binary crossentropy and accuracy, trained with a validation split, and achieved ~81% accuracy when evaluated on the test set. For Iris, which had 3 species labels and 4 features, I defined proper column names, popped the target, and used TensorFlow’s Normalization layer (axis=-1) instead of scikit-learn scalers. The model pipeline started with the normalizer layer, followed by hidden layers, and ended with a Dense(3, softmax) output for multiclass classification. I compiled with Adam optimizer and sparse categorical crossentropy (since labels were integers/strings), trained with validation split, and achieved ~70% accuracy (reasonable given the dataset size). Finally, I saved both models as .keras files.
## Module 3:
### [Convolution Neural-Networks(CNN)](https://github.com/aypy01/tensorflow/blob/main/module-3.ipynb)
> In this module, I worked with convolutional neural networks (CNNs) on three tasks. First, I trained a CNN on the CIFAR-10 dataset, normalizing pixel values to the 0–1 range, stacking Conv2D layers with filters (3×3 kernels) and max pooling, then flattening to connect with dense layers before outputting 10 softmax classes. The model was compiled with Adam and sparse categorical cross-entropy, reaching ~72% accuracy. Next, I repeated the CNN pipeline but applied image augmentation using ImageDataGenerator (zoom, shift, rotation, fill mode), which helped generalize the model but dropped accuracy to ~61%. Finally, I used transfer learning on the Dogs vs Cats dataset with a pretrained MobileNetV2 base model (include_top=False, weights='imagenet'), resizing and normalizing images with 127.0 and then substract 1 from it coz MobileNetV2 needs input in range of -1 to 1 not 0 to 1 before passing them through a GlobalAveragePooling layer and a final dense output. Trained on batched data, this model achieved ~94% accuracy, demonstrating the power of transfer learning for binary classification compared to training a CNN from scratch.

## Module 4 :
### [Reinforcement Learning](https://github.com/aypy01/tensorflow/blob/main/module-4.ipynb)
>  In this module, I implemented Q-learning on the FrozenLake-v1 environment (slippery=True) using Gymnasium. I initialized a Q-table with zeros and defined hyperparameters: learning rate (α), discount factor (γ), exploration rate (ε), episodes, and max steps. At each episode, the environment was reset and actions were chosen using an epsilon-greedy strategy random actions with probability ε, otherwise greedy actions using argmax from the Q-table. After each step, the Q-value was updated using the Bellman equation with the reward and estimated future rewards. Termination or truncation ended an episode, rewards were logged, and ε decayed gradually to favor exploitation. After training, I evaluated performance by averaging rewards over episodes, showing the agent’s ability to improve navigation and achieve ~72% success on FrozenLake.

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
They are saved checkpoints of my experiments — ready to be reloaded for evaluation, predictions, or fine-tuning.

> ### Model Index

| Model File              | Task / Dataset          | Metric Achieved     |
|--------------------------|-------------------------|---------------------|
| `titanic.keras`         | Binary Classification (Titanic survival) | ~81% Accuracy |
| `iris_species.keras`    | Multiclass Classification (Iris dataset) | ~70% Accuracy |
| `cifar10.keras`         | Image Classification (CIFAR-10) | ~72% Accuracy |
| `cifar_augmented.keras` | CIFAR-10 with Data Augmentation | ~61% Accuracy |
| `dogsvscat.keras`       | Transfer Learning (Dogs vs Cats, MobileNetV2) | ~94% Accuracy |
| `fuel_efficiency.keras` | Regression (Auto MPG dataset) | 1.81 MAE |
| `fashion_mnist.keras`   | Image Classification (Fashion MNIST) | ~92.5% Accuracy |

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
> Knowledge should not be gated behind paywalls.
The real edge isn’t in reading endless theory it’s in doing the work, making the mistakes, and watching the feedback loops teach you faster than books ever could. These projects are my proof. Knowledge compounds, like interest, and the only way to collect it is through direct experience. The repo isn’t finishedit never will be. It’s alive, a reflection of progress. If you’re here, don’t just read it. Fork it, break it, rebuild it. That’s how you actually learn.
 This is only a beginning. From preprocessing data to building CNNs and trying out reinforcement learning, each step showed me how much more there is to explore. If you’ve made it this far, don’t stop,turn these foundations into projects, mistakes, and eventually, mastery.
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



