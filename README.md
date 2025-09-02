# TensorFlow Projects by Aditya Yadav
Welcome! This repo is a personal deep learning sandbox These projects mark the beginning of my journey into AI, where I pushed through confusion, code errors, and late-night “why isn't this working?!” moments.

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
>  In this module, I implemented Q-learning on the FrozenLake-v1 environment (slippery=True) using Gymnasium. I initialized a Q-table with zeros and defined hyperparameters: learning rate (α), discount factor (γ), exploration rate (ε), episodes, and max steps. At each episode, the environment was reset and actions were chosen using an epsilon-greedy strategy—random actions with probability ε, otherwise greedy actions using argmax from the Q-table. After each step, the Q-value was updated using the Bellman equation with the reward and estimated future rewards. Termination or truncation ended an episode, rewards were logged, and ε decayed gradually to favor exploitation. After training, I evaluated performance by averaging rewards over episodes, showing the agent’s ability to improve navigation and achieve ~72% success on FrozenLake.

---

## ✨ Projects Included

### 1. 🔤 Sentiment Analysis with TensorFlow & TextVectorization
A complete pipeline that:
- Accepts raw text (like movie reviews) 💬
- Cleans & vectorizes it using `TextVectorization`
- Trains a neural network to predict sentiment (positive/negative) 😃😡
- Outputs predictions on real unseen text

📌 **Key Concepts Covered:**
- Text preprocessing using TensorFlow layers
- Sequential model architecture with Embedding + GlobalAveragePooling
- Binary classification using sigmoid activation
- Evaluation using BinaryCrossentropy & Accuracy metrics
- End-to-end pipeline: raw text → prediction ✅

💡 _Learned how preprocessing must be integrated either before training or as part of the export model, and the importance of thresholding (e.g., 0.5 for binary)._

---

### 2. 🖼️ Basic Image Classification (CNN)
My early dive into image data:
- Built a basic **Convolutional Neural Network** (CNN)
- Used TensorFlow’s `ImageDataGenerator` for loading and augmenting image data
- Trained the model to classify images into categories 🎯

📌 **Key Concepts Covered:**
- Convolution and MaxPooling operations
- Flattening image features into dense layers
- Model evaluation on validation/test sets
- Use of `compile`, `fit`, and `evaluate` pipelines
- Understanding of overfitting and how to reduce it (e.g., via dropout or data augmentation)

💡 _This project was especially useful for getting hands-on with CNNs and understanding how image features are extracted._

---

## 🚀 Personal Note

When I started, I thought deep learning would be out of my league. Reading through TensorFlow docs felt like reading alien symbols at first 👽. But with some persistence, trial and error, and targeted help from ChatGPT — I began to not just copy code, but understand it. This repo stands as proof that **you can go from zero to building real ML pipelines by learning actively and not giving up.**

I did **everything here by myself**, without copying from others, relying only on official docs, Google searches, and intelligent nudges from ChatGPT when I got stuck. This was all **me vs the machine**… and I think I’m starting to win 😄.

---

## 🛠 Tech Stack
- TensorFlow / Keras
- Python 3
- Google Colab



---
## Note
Knowledge should not be gated behind paywalls or exclusivity.
This repository exists so that anyone can access structured, practical TensorFlow notes without restriction.
The journey doesn’t end here. After mastering these core modules data preprocessing, model building, CNNs, and reinforcement learning—take the next step with fullfledged TensorFlow projects that put theory into real-world practice.

Thanks for reading. And if you’re just starting your ML journey — **don’t stop now. You’re one project away from your breakthrough.** 

This is only a beginning. From preprocessing data to building CNNs and trying out reinforcement learning, each step showed me how much more there is to explore. If you’ve made it this far, don’t stop—turn these foundations into projects, mistakes, and eventually, mastery.

## Contact Me
Feel free to drop feedback or collaboration ideas:
-  [🧠 GitHub ](https://github.com/aditya-yadav-ai) 


