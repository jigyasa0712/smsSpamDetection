import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk_data_path = './nltk_data'
nltk.data.path.append(nltk_data_path)

# Check and download 'punkt'
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' NLTK data...")
    nltk.download('punkt', download_dir=nltk_data_path)
    print("'punkt' NLTK data downloaded.")

# Check and download 'stopwords'
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' NLTK data...")
    nltk.download('stopwords', download_dir=nltk_data_path)
    print("'stopwords' NLTK data downloaded.")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Detection")
input_sms = st.text_area("Enter Your Message")

if st.button('Predict'):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result==1 :
        st.header("Spam")
    else:
        st.header("Not Spam")