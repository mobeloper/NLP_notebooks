
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
model = joblib.load('review_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess the text similar to how the model was trained
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    return text

# Streamlit App UI
st.title("Customer Review Classifier")

# Text input
user_input = st.text_input("Enter a business review:")

if st.button("Classify"):
    if user_input:
        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Vectorize the input text
        input_tfidf = vectorizer.transform([processed_input])

        # Get the prediction
        prediction = model.predict(input_tfidf)

        # Output result
        if prediction[0] == 1:
            st.write("This is a **HAPPY** review.")
        else:
            st.write("This is a **SAD** review.")
    else:
        st.write("Please enter a review to classify.")
