import streamlit as st
import pickle
import pandas as pd
import numpy as np
import preprocess_kgptalkie as ps
import contractions
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK resources (you only need to do this once)
# nltk.download('punkt')
# nltk.download('stopwords')


# *******FUNCTIONS***********

def processing_func(text):
    res=contractions.fix(text)
    return res

import string
def remove_punctuation(text):
    regular_punct = list(string.punctuation)
    for punc in regular_punct:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()

def stem_text(text):
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    
    # Tokenize the text (split it into words)
    words = nltk.word_tokenize(text)
    
    # Apply stemming to each word and remove stopwords
    stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stopwords.words('english')]
    
    # Join the stemmed words back into a sentence
    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text

import itertools
def remove_duplicates_car(tweet):
    new_tweet = ''.join(ch for ch, _ in itertools.groupby(tweet))
    return new_tweet

# *******************************

# load model
model = pickle.load(open('sentiment_analysis.pkl', 'rb'))
# Set page title and description
st.markdown("<h1 style='text-align: center; color: #cdaf98;'>IMDB Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("<h4 style='text-align: left; color: #eee5dd; '> Write your sentiment: </h4>", unsafe_allow_html=True)

# Create an input field
review = st.text_area('', height=200)




# Preprocessing

review = ps.remove_stopwords(review)

# Remove HTML tags (assuming you have a function called remove_html_tags)
review = ps.remove_html_tags(review)

# Convert text to lowercase
review = review.lower()

# Remove punctuation (assuming you have a function called remove_punctuation)
review = remove_punctuation(review)

# Apply processing function (assuming you have a function called processing_func)
review = processing_func(review)

# Apply stemming function (assuming you have a function called stem_text)
review = stem_text(review)

# Apply remove duplicates function (assuming you have a function called remove_duplicates_car)
review = remove_duplicates_car(review)



# Add a submit button
prediction = model.predict([review])[0]
submit = st.button('Predict')
if submit:
    if not review.strip():  # Check if the input is empty
        st.warning('Please enter a real review.')
    elif prediction =='positive':
        
        
        st.success('Positive Review')   
    else:
        
        
      st.warning('Negative Review')
      
      
      
      #streamlit run app.py