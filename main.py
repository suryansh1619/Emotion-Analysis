import streamlit as st
from model import predict_emotion

st.title("Emotion Analysis App ")

user_input = st.text_area("Enter a sentence:", "")

if st.button("Analyze Emotion"):
    if user_input.strip():
        predicted_emotion = predict_emotion(user_input)
        st.write(f"**Predicted Emotion:** {predicted_emotion}")  
    else:
        st.warning("Please enter some text to analyze.")


