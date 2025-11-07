import app as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("gru_mode.h5")
    with open(r"C:\Users\nitin\code\tokenizer (1).pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model()
total_words = len(tokenizer.word_index) + 1

def predict_next_word(model, tokenizer, text, top_k=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=20, padding='pre')

    preds = model.predict(sequence, verbose=0)[0]
    sorted_indices = np.argsort(preds)[-top_k:][::-1]

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    predicted_words = [index_to_word.get(i, '') for i in sorted_indices]
    return predicted_words

st.title("🧠 Next Word Prediction App")
st.write("Type a few words and let the model predict what comes next!")

user_input = st.text_input("Enter a sentence:", "")

if st.button("🔮 Predict Next Word"):
    if user_input.strip():
        predictions = predict_next_word(model, tokenizer, user_input)
        st.success(f"Top Predictions: {', '.join(predictions)}")
    else:
        st.warning("Please enter some text first!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
