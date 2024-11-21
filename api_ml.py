from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
from utils import (
    emoji_dict,
    stop_words,
    get_top_3_positive_comments,
    get_top_3_negative_comments,
)
from nltk.tokenize import RegexpTokenizer
import re
from utils import get_top_3_negative_comments, get_top_3_positive_comments

app = FastAPI()

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = load_model("model/bert_attention.h5")

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
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stop_words_removal(text)
    return text

def tokenize_batch(texts):
    return tokenizer(
        texts, padding="max_length", truncation=True, max_length=128, return_tensors="tf"
    )['input_ids']

def predict_sentiment_batch(texts):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    tokenized_texts = tokenize_batch(preprocessed_texts)
    predictions = model.predict(tokenized_texts)
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    sentiments = [sentiment_labels[pred.argmax()] for pred in predictions]
    return predictions, sentiments

# Input schema
class Comment(BaseModel):
    text: str

class RequestBody(BaseModel):
    comments: List[Comment]

@app.post("/predict")
async def predict_sentiments(data: RequestBody):
    # Extract text from comments
    texts = [comment.text for comment in data.comments]
    
    # Predict sentiments and probabilities
    predictions, sentiments = predict_sentiment_batch(texts)
    
    # Summarize sentiments
    summary = {
        "Positif": sentiments.count("Positif"),
        "Netral": sentiments.count("Netral"),
        "Negatif": sentiments.count("Negatif"),
    }
    
    top_3_positive = get_top_3_positive_comments(predictions, texts)
    top_3_negative = get_top_3_negative_comments(predictions, texts)
    
    # Return results
    return {
        "sentiments": sentiments,
        "summary": summary,
        "top_3_positive": top_3_positive,
        "top_3_negative": top_3_negative,
    }
