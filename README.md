# TensorFlow Projects by Aditya Yadav
Welcome! This repo is a personal deep learning sandbox where I, **Aditya Yadav**, explore machine learning from scratch — guided only by documentation, curiosity, and some much-needed help from ChatGPT 😅. These projects mark the beginning of my journey into AI, where I pushed through confusion, code errors, and late-night “why isn't this working?!” moments.

## Module 1:

> Module 1 mainly circled around the basics of TensorFlow vs NumPy, the idea of tensors being immutable (like tuples) and tf.Variable being mutable (needed for weights that update), slicing feeling familiar yet tricky because reassignment only works with variables, and the bigger gap you noticed in your own understanding: you’ve practiced shapes but never really used reshape for a real purpose — and now you know it matters only when data has to be bent into the exact form a model expects, not for its own sake. You also clarified that messy or unbalanced data isn’t a reason to “go deep learning” — that’s preprocessing work, not architecture choice.
## Module 2:

> Worked with two datasets from TensorFlow: Titanic (binary classification) and Iris (multiclass classification). For Titanic, I learned to properly split train/test sets (x_train, y_train, x_test, y_test) and handled missing values with fillna. I dropped the fare column (not useful for survival), scaled numerical features (age, etc.) with StandardScaler, and one-hot encoded categorical columns. After preprocessing, I built a neural network with input shape = number of features, hidden layers, and a single sigmoid output unit for binary classification. The model was compiled with binary crossentropy and accuracy, trained with a validation split, and achieved ~81% accuracy when evaluated on the test set. For Iris, which had 3 species labels and 4 features, I defined proper column names, popped the target, and used TensorFlow’s Normalization layer (axis=-1) instead of scikit-learn scalers. The model pipeline started with the normalizer layer, followed by hidden layers, and ended with a Dense(3, softmax) output for multiclass classification. I compiled with Adam optimizer and sparse categorical crossentropy (since labels were integers/strings), trained with validation split, and achieved ~70% accuracy (reasonable given the dataset size). Finally, I saved both models as .keras files. 

---

Welcome! This repo is a personal deep learning sandbox where I, **Aditya Yadav**, explore machine learning from scratch — guided only by documentation, curiosity, and some much-needed help from ChatGPT 😅. These projects mark the beginning of my journey into AI, where I pushed through confusion, code errors, and late-night “why isn't this working?!” moments.

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
- Markdown (for this cool README 😎)

---

## 📌 Next Steps
- Clean up and polish the code
- Host sentiment model as an API using Flask/FastAPI
- Try fine-tuning BERT or another pre-trained model
- Build more advanced image models (ResNet, etc.)
- Explore multi-class classification

---

## 👀 Demo Coming Soon?
Might drop a streamlit web app or a simple HuggingFace space… stay tuned!

---

## 📫 Contact Me
Feel free to drop feedback or collaboration ideas:
-  [🧠 GitHub ](https://github.com/aditya-yadav-ai) 
---

Thanks for reading. And if you’re just starting your ML journey — **don’t stop now. You’re one project away from your breakthrough.** 🚀
