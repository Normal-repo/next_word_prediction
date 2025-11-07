# 🧠 Next Word Prediction App (GRU + TensorFlow)

A **Streamlit web app** that predicts the **next word** in a given sentence using a **GRU (Gated Recurrent Unit)** neural network trained on text data with **TensorFlow/Keras**.

---

## 🚀 Demo

Predict the next word in real-time!  
Type a few words, and the model suggests the top likely completions.

![App Screenshot](https://raw.githubusercontent.com/Normal-repo/next_word_prediction/main/screenshot.png)

*(Add a screenshot of your app once it’s running locally or deployed.)*

---

## 🧩 Project Overview

This project combines **deep learning** (GRU model) with an easy-to-use **Streamlit interface** for next-word prediction.

### 🔹 Core Components
- `app.py` – Streamlit web app for prediction  
- `gru_mode.h5` – Pre-trained GRU model  
- `tokenizer.pkl` – Tokenizer for sequence generation  
- `train_gru_model.ipynb` – Notebook used for training  
- `requirements.txt` – Dependency list  

---

## ⚙️ How It Works

1. User enters a partial sentence in the Streamlit interface.  
2. The tokenizer converts text into integer sequences.  
3. The sequence is padded to match the model’s training length.  
4. The GRU model predicts the probability of the next word.  
5. The app displays the **top 3 predictions** ranked by probability.

---

## 🧠 Model Details

| Property | Description |
|-----------|--------------|
| **Model Type** | GRU (Recurrent Neural Network) |
| **Framework** | TensorFlow / Keras |
| **Input** | Tokenized text sequences |
| **Output** | Next-word prediction (softmax probabilities) |
| **Saved Model File** | `gru_mode.h5` |
| **Tokenizer File** | `tokenizer.pkl` |

---

## 🧪 Model Training Notebook

The GRU model used in this app was trained using:

📘 [`train_gru_model.ipynb`](https://github.com/Normal-repo/next_word_prediction/blob/main/train_gru_model.ipynb)

### Notebook Highlights
- Text preprocessing and tokenization  
- Sequence generation for next-word prediction  
- GRU layers with embedding  
- Trained with categorical cross-entropy loss  
- Saved trained model and tokenizer for reuse  

---
