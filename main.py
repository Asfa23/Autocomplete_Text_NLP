import streamlit as st
import pickle
import string
import nltk
from nltk.util import ngrams
from collections import Counter
import math
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# A. Preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s.,;!?]', '', text.lower())
    return text

# B. Tokenization
def tokenize(text):
    return nltk.word_tokenize(text)

# C. Remove stop words
def remove_stopwords(tokens):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# D. Perform lemmatization
def perform_lemmatization(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# E. Build n-gram model and calculate probabilities
def build_ngram_model(tokens, n):
    ngrams_list = ngrams(tokens, n)
    ngram_counts = Counter(ngrams_list)
    total_ngrams = sum(ngram_counts.values())
    ngram_probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return ngram_probabilities

# F. Simple probability for prediction
def simple_prob_predict_next_word(input_text, ngram_probabilities, n):
    tokens = preprocess_text(input_text)
    tokens = tokenize(tokens)
    tokens = remove_stopwords(tokens)
    tokens = perform_lemmatization(tokens)
    ngram_prefix = tuple(tokens[-(n-1):])

    predictions = []
    for ngram, prob in ngram_probabilities.items():
        if ngram[:n-1] == ngram_prefix:
            predictions.append((ngram[-1], prob))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# G. Bayesian prediction
def bayesian_predict_next_word(input_text, ngram_probabilities, n):
    tokens = preprocess_text(input_text)
    tokens = tokenize(tokens)
    tokens = remove_stopwords(tokens)
    tokens = perform_lemmatization(tokens)
    ngram_prefix = tuple(tokens[-(n-1):])

    # Filter n-grams that match the prefix
    relevant_ngrams = {ngram: prob for ngram, prob in ngram_probabilities.items() if ngram[:n-1] == ngram_prefix}

    # Calculate the denominator for Bayesian probability
    denominator = sum(relevant_ngrams.values())

    # Calculate Bayesian probabilities
    predictions = [(ngram[-1], (prob + 1) / (denominator + len(ngram_probabilities))) for ngram, prob in relevant_ngrams.items()]

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


# I. Fungsi untuk mendapatkan hasil berupa teks dengan kata-kata saja
def get_formatted_predictions(predictions, max_words):
    # Ambil hanya sejumlah maksimum kata yang diinginkan
    predictions = predictions[:max_words]

    # Ambil kata-kata saja
    formatted_predictions = [word for word, prob in predictions]

    # Gabungkan hasil menjadi kalimat
    result = ' '.join(formatted_predictions)
    return result

# Memuat objek n-gram probabilitas dari file menggunakan pickle
with open('ngram_probabilities.pkl', 'rb') as f:
    ngram_probabilities = pickle.load(f)

# Input text for prediction
input_text = "my parent told me to eat"
# Set the maximum number of words to display in predictions
max_words = 5

# Perform predictions using simple probability
simple_prob_predictions = simple_prob_predict_next_word(input_text, ngram_probabilities, n=3)
simple_prob_output = get_formatted_predictions(simple_prob_predictions, max_words)

# Perform predictions using Bayesian approach
bayesian_predictions = bayesian_predict_next_word(input_text, ngram_probabilities, n=3)
bayesian_output = get_formatted_predictions(bayesian_predictions, max_words)

def main():
    st.title("Autocomplete Text")
    
    # Input text for prediction
    input_text = st.text_input("Masukkan Kata Bertemakan 'Food' dalam Bahasa Inggris :")
    
    # Set the maximum number of words to display in predictions
    max_words = st.number_input("Number of words to predict:", min_value=1, value=5)

    if st.button("Predict"):
        # Perform predictions using simple probability
        simple_prob_predictions = simple_prob_predict_next_word(input_text, ngram_probabilities, n=3)
        simple_prob_output = get_formatted_predictions(simple_prob_predictions, max_words)

        # Perform predictions using Bayesian approach
        bayesian_predictions = bayesian_predict_next_word(input_text, ngram_probabilities, n=3)
        bayesian_output = get_formatted_predictions(bayesian_predictions, max_words)

        st.write("\nSimple Probability Predictions:")
        st.write(input_text, simple_prob_output)

        st.write("\nBayesian Predictions:")
        st.write(input_text, bayesian_output)

if __name__ == "__main__":
    main()