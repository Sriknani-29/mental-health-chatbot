import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from gtts import gTTS
import os

# Load model and tokenizer
model_path = "./emotion_model"
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

label_map = {
    0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"
}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_map[prediction]

def generate_response(emotion):
    responses = {
        "joy": "That's wonderful! Keep smiling.",
        "sadness": "I'm really sorry you're feeling down. I'm here for you.",
        "anger": "I understand you're upset. Let's breathe and talk it out.",
        "fear": "That sounds scary. You're not alone—I'm with you.",
        "love": "Love is beautiful! It's great to feel connected.",
        "surprise": "Oh wow! That sounds unexpected. Want to share more?"
    }
    return responses.get(emotion, "Thanks for sharing. I'm here for you.")

def speak_response(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    return "response.mp3"

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot")
st.markdown("Type your thoughts below and let me support you emotionally.")

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Get Support"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        response = generate_response(emotion)

        st.subheader("🧾 Detected Emotion:")
        st.success(emotion.capitalize())

        st.subheader("💬 Chatbot Response:")
        st.info(response)

        # Optional audio response
        audio_file = speak_response(response)
        audio_bytes = open(audio_file, 'rb').read()
        st.audio(audio_bytes, format='audio/mp3')
    else:
        st.warning("Please enter some text first.")
