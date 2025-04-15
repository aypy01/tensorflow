# 🧠 TensorFlow Projects by Aditya Yadav

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
- ✉️ Email: Coming soon
- 🧠 GitHub: [github.com/aditya-yadav-ai](https://github.com/aditya-yadav-ai) *(you can update with actual username)*

---

Thanks for reading. And if you’re just starting your ML journey — **don’t stop now. You’re one project away from your breakthrough.** 🚀
