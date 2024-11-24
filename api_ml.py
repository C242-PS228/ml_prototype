from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
from utils import (
    emoji_dict,
    stop_words,
    get_top_3_positive_comments,
    get_top_3_negative_comments,
    get_top_3_common_words,
    analyze_sentiment_top_3
)
from nltk.tokenize import RegexpTokenizer
import re
# from collections import Counter
# import stanza
import numpy as np

app = FastAPI()

# Load the model, tokenizer, and NLP pipeline
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = load_model("model/bert_attention.h5")
# custom_dir = "./stanza_models"
# nlp = stanza.Pipeline('id', dir=custom_dir)

# Preprocessing functions
def replace_emoji_with_word(text):
    for emoji, word in emoji_dict.items():
        text = re.sub(f'({emoji})+', f' {word} ', text)
    return text.strip()

def stop_words_removal(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    removed_stop_words = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(removed_stop_words)

def preprocess_text(text):
    text = text.lower()
    text = replace_emoji_with_word(text)
    text = re.sub(r'@\w+', '', text).strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stop_words_removal(text)
    return text

def tokenize_batch(texts):
    return tokenizer(
        texts, padding="max_length", truncation=True, max_length=128, return_tensors="tf"
    )['input_ids']

# def predict_sentiment_batch(texts):
#     preprocessed_texts = [preprocess_text(text) for text in texts]
#     tokenized_texts = tokenize_batch(preprocessed_texts)
#     predictions = model.predict(tokenized_texts)
#     sentiment_labels = ["Negatif", "Netral", "Positif"]
#     sentiments = [sentiment_labels[pred.argmax()] for pred in predictions]
#     return predictions, sentiments

def predict_sentiment_batch(texts, preprocess=True):
    if preprocess:
        preprocessed_texts = [preprocess_text(text) for text in texts]
    else:
        preprocessed_texts = texts
    tokenized_texts = tokenize_batch(preprocessed_texts)
    predictions = model.predict(tokenized_texts)
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    sentiments = [sentiment_labels[pred.argmax()] for pred in predictions]
    class_labels = np.argmax(predictions, axis=1)
    return sentiments, class_labels, predictions

# Input schema
class Comment(BaseModel):
    text: str

class RequestBody(BaseModel):
    comments: List[Comment]

@app.post("/predict")
async def predict_sentiments(data: RequestBody):
    # Extract text from comments
    texts = [comment.text for comment in data.comments]
    preprocessed_texts = [preprocess_text(text) for text in texts]
    # Predict sentiments and probabilities
    sentiments, class_labels, predictions = predict_sentiment_batch(preprocessed_texts, preprocess=False)
    
    # Summarize sentiments
    summary = {
        "Positif": sentiments.count("Positif"),
        "Netral": sentiments.count("Netral"),
        "Negatif": sentiments.count("Negatif"),
    }
    
    # Get top 3 positive and negative comments
    top_3_positive = get_top_3_positive_comments(predictions, texts)
    top_3_negative = get_top_3_negative_comments(predictions, texts)
    # Analyze the most common positive and negative words
    pos_common_words, neg_common_words = analyze_sentiment_top_3(preprocessed_texts, class_labels)
    
    # Return results
    return {
        "sentiments": sentiments,
        "summary": summary,
        "top_3_positive": top_3_positive,
        "top_3_negative": top_3_negative,
        "top_3_pos_words": pos_common_words,
        "top_3_neg_words": neg_common_words
    }
