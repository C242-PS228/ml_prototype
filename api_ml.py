from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import utils

app = FastAPI()

# Load the model, tokenizer, and NLP pipeline
tokenizer = utils.load_tokenizer("tokenizer")
model = utils.load_nlp_model("model/bert_attention_v6.h5")

nlp = utils.load_stanza_pipeline()

# Input schema
class Comment(BaseModel):
    text: str

class RequestBody(BaseModel):
    comments: List[Comment]

@app.post("/predict")
async def predict_sentiments(data: RequestBody):
    # Extract text from comments
    texts = [comment.text for comment in data.comments]
    preprocessed_texts = [utils.preprocess_text(text) for text in texts]
    
    # Predict sentiments and probabilities
    sentiments, class_labels, predictions = utils.predict_sentiment_batch(preprocessed_texts, model=model, tokenizer=tokenizer, preprocess=False)
    
    # Summarize sentiments
    positive_count = sentiments.count("Positif")
    neutral_count = sentiments.count("Netral")
    negative_count = sentiments.count("Negatif")
    
    # Get top 3 positive and negative comments
    top_3_positive = utils.get_top_3_positive_comments(predictions, texts)
    top_3_negative = utils.get_top_3_negative_comments(predictions, texts)
    
    # Analyze the most common positive and negative key_words
    liked_by_cust, disliked_by_cust = utils.get_key_words_and_clean_up(preprocessed_texts, class_labels, stanza=nlp)
    
    # Convert key_words to the desired format
    positive_key_words = [{"tagname": tag, "value": count} for tag, count in liked_by_cust.items()]
    negative_key_words = [{"tagname": tag, "value": count} for tag, count in disliked_by_cust.items()]
    
    # Return results in the new format
    return {
        "data": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "topstatus": {
                "positive": top_3_positive,
                "negative": top_3_negative
            },
            "key_words": {
                "positive": positive_key_words,
                "negative": negative_key_words
            }
        }
    }