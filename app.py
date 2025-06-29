import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Define STOP_WORDS globally
STOP_WORDS = set(stopwords.words("english"))

# Load the pre-trained model and Word2Vec model
@st.cache_resource
def load_model_and_word2vec():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("word2vec_model.pkl", "rb") as file:
        word2vec_model = pickle.load(file)
    return model, word2vec_model

# Preprocess a single question
def preprocess_question(question):
    """
    Preprocess a single question by cleaning, normalizing, and transforming the text.
    """
    # Convert to lowercase and strip whitespace
    question = question.lower().strip()

    # Replace special characters with their string equivalents
    replacements = {
        '%': ' percent ', '$': ' dollar ', '₹': ' rupee ', '€': ' euro ', '@': ' at ', '[math]': ''
    }
    for key, val in replacements.items():
        question = question.replace(key, val)

    # Replace large numbers with their word equivalents
    large_number_replacements = {
        r'(\d+)000000000000': r'\1 trillion',
        r'(\d+)000000000': r'\1 billion',
        r'(\d+)000000': r'\1 million',
        r'(\d+)000': r'\1 thousand'
    }
    for key, val in large_number_replacements.items():
        question = re.sub(key, val, question)

    # Expand contractions
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
        "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
        "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
        "he's": "he is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
        "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
        "it's": "it is", "it'd": "it would", "it'd've": "it would have", "let's": "let us",
        "ma'am": "madam", "mustn't": "must not", "needn't": "need not", "o'clock": "of the clock",
        "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not", "should've": "should have",
        "shouldn't": "should not", "shouldn't've": "should not have", "that'd": "that would",
        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
        "there'd've": "there would have", "there's": "there is", "they'd": "they would",
        "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
        "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would",
        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
        "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
        "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did",
        "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
        "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
        "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
        "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
        "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
        "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",
        "'cause": "because", "'til": "until", "'tis": "it is", "'twas": "it was", "'d": " would",
        "'ll": " will", "'re": " are", "'ve": " have", "n't": " not"
    }
    question = ' '.join([contractions[word] if word in contractions else word for word in question.split()])

    # Remove HTML tags
    question = BeautifulSoup(question, "html.parser").get_text()

    # Remove punctuation and extra spaces
    question = re.sub(r'\W+', ' ', question).strip()

    return question

# Function to compute additional features
def compute_additional_features(question1, question2):
    """
    Compute the 21 additional features for the pair of questions.
    """
    # Word count features
    q1_len = len(question1.split())
    q2_len = len(question2.split())
    abs_len_diff = abs(q1_len - q2_len)

    # Common words
    q1_words = set(question1.split())
    q2_words = set(question2.split())
    common_words = len(q1_words.intersection(q2_words))

    # Fuzzy matching scores
    fuzz_ratio = fuzz.ratio(question1, question2)
    fuzz_partial_ratio = fuzz.partial_ratio(question1, question2)
    token_sort_ratio = fuzz.token_sort_ratio(question1, question2)
    token_set_ratio = fuzz.token_set_ratio(question1, question2)

    # Token features
    q1_tokens = question1.split()
    q2_tokens = question2.split()
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    common_word_count = len(q1_words.intersection(q2_words))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    cwc_min = common_word_count / (min(len(q1_words), len(q2_words)) + 0.0001)
    cwc_max = common_word_count / (max(len(q1_words), len(q2_words)) + 0.0001)
    ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + 0.0001)
    ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + 0.0001)

    # Check if questions have tokens before accessing first/last token
    last_word_eq = int(q1_tokens[-1] == q2_tokens[-1]) if q1_tokens and q2_tokens else 0
    first_word_eq = int(q1_tokens[0] == q2_tokens[0]) if q1_tokens and q2_tokens else 0

    # Length-based features
    longest_substr = list(distance.lcsubstrings(question1, question2))
    longest_substr_length = len(longest_substr[0]) if longest_substr else 0
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
    longest_substr_ratio = longest_substr_length / (min(len(question1), len(question2)) + 1)

    # Skimming feature
    skimming_feature = len(set(q1_tokens).intersection(set(q2_tokens))) / (min(len(q1_tokens), len(q2_tokens)) + 1)

    # Word share
    total_words = len(q1_words) + len(q2_words)
    word_share = common_words / total_words if total_words > 0 else 0

    # Total words
    total_words_feature = total_words

    # Additional features
    q1_new_word = len(question1.split())
    q2_new_word = len(question2.split())

    # Combine all features into a single array
    features = np.array([
        q1_len, q2_len, abs_len_diff, common_words, fuzz_ratio, fuzz_partial_ratio,
        token_sort_ratio, token_set_ratio, cwc_min, cwc_max, ctc_min, ctc_max,
        last_word_eq, first_word_eq, mean_len, longest_substr_ratio, skimming_feature,
        word_share, total_words_feature, q1_new_word, q2_new_word
    ])
    return features

# Function to compute average Word2Vec embedding for a question
def get_avg_word2vec(question, model):
    vectors = [model.wv[word] for word in question.split() if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Streamlit app
def main():
    st.title("Question Duplicate Detection")
    st.write("Enter two questions")

    # Input fields for questions
    question1 = st.text_input("Enter Question 1:")
    question2 = st.text_input("Enter Question 2:")

    if st.button("Check Duplicate"):
        if question1 and question2:
            # Preprocess the questions
            question1 = preprocess_question(question1)
            question2 = preprocess_question(question2)

            # Load the model and Word2Vec model
            model, word2vec_model = load_model_and_word2vec()

            # Compute Word2Vec embeddings
            q1_word2vec = get_avg_word2vec(question1, word2vec_model)
            q2_word2vec = get_avg_word2vec(question2, word2vec_model)

            # Compute additional features
            additional_features = compute_additional_features(question1, question2)

            # Combine Word2Vec embeddings with additional features
            combined_features = np.hstack((q1_word2vec, q2_word2vec, additional_features))

            # Predict using the model
            prediction_prob = model.predict_proba([combined_features])
            prediction = model.predict([combined_features])

            # Convert prediction to human-readable format
            result = "Duplicate" if prediction[0] == 1 else "Not Duplicate"

            # Display the result
            st.success(f"Result: {result} ")
        else:
            st.warning("Please enter both questions.")

if __name__ == "__main__":
    main()