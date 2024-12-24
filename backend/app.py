from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set the NLTK data path
nltk.data.path.append('D:/PROJECTS/AIML/FAKE JOB DETECTION SYSTEM/nltk_data')

# Download the necessary NLTK data
nltk.download('stopwords', download_dir='D:/PROJECTS/AIML/FAKE JOB DETECTION SYSTEM/nltk_data')
nltk.download('punkt', download_dir='D:/PROJECTS/AIML/FAKE JOB DETECTION SYSTEM/nltk_data')

app = Flask(__name__)

# Load the saved model and vectorizer
model_path = 'D:/PROJECTS/AIML/FAKE JOB DETECTION SYSTEM/fake_job_detection_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

vectorizer_path = 'D:/PROJECTS/AIML/FAKE JOB DETECTION SYSTEM/tfidf_vectorizer.pkl'
if os.path.exists(vectorizer_path):
    with open(vectorizer_path, 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)
else:
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in tokens if word not in stop_words])
    return text

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get('title', '')
    description = data.get('description', '')

    # Combine title and description and preprocess
    text = preprocess_text(title + " " + description)
    text_vector = tfidf.transform([text]).toarray()

    # Predict using the model
    prediction = model.predict(text_vector)
    response = {
        'prediction': 'Fake' if prediction[0] == 1 else 'Real'
    }
    return jsonify(response)

# Home route
@app.route('/')
def home():
    return "<h1>Fake Job Detection System API is running!</h1>"

if __name__ == '__main__':
    app.run(debug=True)
