import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load models
baseline_model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
lstm_model = load_model("models/lstm_model.h5")

# Tokenizer for LSTM
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

st.title("📊 Sentiment Analysis App")
st.write("Classify text into **positive**, **neutral**, or **negative**.")

user_input = st.text_area("Enter text:", "")

if st.button("Predict"):
    if user_input.strip() != "":
        # Baseline prediction
        X_vec = vectorizer.transform([user_input])
        baseline_pred = baseline_model.predict(X_vec)[0]
        baseline_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        # LSTM prediction
        X_seq = tokenizer.texts_to_sequences([user_input])
        X_pad = pad_sequences(X_seq, maxlen=50, padding="post")
        lstm_pred = lstm_model.predict(X_pad).argmax(axis=1)[0]
        lstm_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        st.subheader("🔹 Results")
        st.write(f"**Baseline (LogReg):** {baseline_map[baseline_pred]}")
        st.write(f"**Deep Learning (LSTM):** {lstm_map[lstm_pred]}")
