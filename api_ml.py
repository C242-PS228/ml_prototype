from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import utils

app = FastAPI()

# Load the model, tokenizer, and NLP pipeline
tokenizer = utils.load_tokenizer("tokenizer")
model = utils.load_nlp_model("model/bert_attention.h5")

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
    summary = {
        "Positif": sentiments.count("Positif"),
        "Netral": sentiments.count("Netral"),
        "Negatif": sentiments.count("Negatif"),
    }
    
    # Get top 3 positive and negative comments
    top_3_positive = utils.get_top_3_positive_comments(predictions, texts)
    top_3_negative = utils.get_top_3_negative_comments(predictions, texts)
    # Analyze the most common positive and negative words
    pos_common_words, neg_common_words = utils.analyze_sentiment_top_3(preprocessed_texts, class_labels, stanza=nlp)
    
    # Return results
    return {
        "sentiments": sentiments,
        "summary": summary,
        "top_3_positive": top_3_positive,
        "top_3_negative": top_3_negative,
        "top_3_pos_words": pos_common_words,
        "top_3_neg_words": neg_common_words
    }
