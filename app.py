import nltk
import string
import re
import streamlit as st
import sklearn
import joblib

def lower(text):
    return text.lower()

from bs4 import BeautifulSoup
def html_tag(text):
  soup = BeautifulSoup(text,"html.parser")
  new_text = soup.get_text()
  return new_text

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    filtered_text = ' '.join(filtered_text)
    return filtered_text

nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()
def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    stems = ' '.join(stems)
    return stems
  
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
def lemmatize_word(text):
  word_tokens = word_tokenize(text)
  lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
  lemmas = ' '.join(lemmas)
  return lemmas  

def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

model = joblib.load('SentimentAnalysis_model')
st.title('Sentiment Analyzer')
ip = st.text_input("Enter The Message")
ip = lower(ip)
ip = remove_punctuation(ip)
ip = remove_numbers(ip)
ip = html_tag(ip)
ip = stem_words(ip)
ip = lemmatize_word(ip)
ip = remove_stopwords(ip)
op = model.predict([ip])
if st.button("Predict"):
  st.title(op[0]) 
