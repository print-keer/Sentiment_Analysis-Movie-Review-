import pandas as pd
import pickle as pk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
# model_path = r'C:\Users\keerthi\OneDrive\Desktop\Sentiment Analysis (movie review)\model1.pkl'
model = pk.load(open('model.pkl', 'rb'))
# skaler_path = r'C:\Users\keerthi\OneDrive\Desktop\Sentiment Analysis (movie review)\scaler1.pkl'
scaler = pk.load(open('scaler.pkl', 'rb'))

# Streamlit app
st.title("Movie Review Sentiment Analysis")

name = st.text_input("Enter The Movie Name :")

review = st.text_input("Enter Movie Review")


if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write("You gave a negative review :(")
    else:
        st.write("You gave a positive review :)")