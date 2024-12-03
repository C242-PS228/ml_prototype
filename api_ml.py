from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import utils


app = FastAPI()

# Load the model, tokenizer, and NLP pipeline
tokenizer = utils.load_tokenizer("tokenizer")
model = utils.load_nlp_model("https://storage.googleapis.com/production_model_storage/perceivo_models/bert_attention_v11.h5")
question_model = utils.load_nlp_model("model/question_bert.h5")
assistance_model = utils.load_nlp_model("model/assistance_bert_v2.h5")

nlp = utils.load_stanza_pipeline()

class Comment(BaseModel):
    text: Optional[str] = Field(None, description="Comment text can be null or missing.")
    username: Optional[str] = Field(None, description="Username of the commenter.")


class RequestBody(BaseModel):
    comments: List[Comment]

@app.post("/predict")
async def predict_sentiments(data: RequestBody):
    texts = []
    usernames = []
    for comment in data.comments:
        if comment.text is not None:
            texts.append(comment.text.strip())
            usernames.append(comment.username)
    texts_to_index_map = {text: idx for idx, text in enumerate(texts)}
    # texts = [comment.text for comment in data.comments if comment.text is not None and comment.text.strip()]
    # usernames = [comment.username for comment in data.comments if comment.text and comment.text.strip()]

    if not texts:
        raise HTTPException(status_code=400, detail="All comments are null, empty, or missing text.")

    preprocessed_texts = [utils.preprocess_text(text) for text in texts]

    # print(preprocessed_texts)
    
    # Predict sentiments and probabilities
    sentiments, class_labels, predictions = utils.predict_sentiment_batch(preprocessed_texts, model=model, tokenizer=tokenizer, preprocess=False)
    
    # Summarize sentiments
    positive_count = sentiments.count("Positif")
    neutral_count = sentiments.count("Netral")
    negative_count = sentiments.count("Negatif")
    
    # Get top 3 positive and negative comments
    top_3_positive = utils.get_top_3_positive_comments(predictions, texts)
    top_3_negative = utils.get_top_3_negative_comments(predictions, texts)
    # Get the username of the top comments
    top_3_pos_username = utils.get_username(texts_to_index_map, top_3_positive, usernames)
    top_3_neg_username = utils.get_username(texts_to_index_map, top_3_negative, usernames)
    top_3_pos_comments_username = [{"username": top_3_pos_username[i], "text": top_3_positive[i]} for i in range(len(top_3_positive))]
    top_3_neg_comments_username = [{"username": top_3_neg_username[i], "text": top_3_negative[i]} for i in range(len(top_3_negative))]
        
    # Analyze the most common positive and negative key_words
    liked_by_cust, disliked_by_cust = utils.get_key_words_and_clean_up(preprocessed_texts, class_labels, stanza=nlp, model=model, tokenizer=tokenizer)
    
    # Convert key_words to the desired format
    positive_key_words = [{"tagname": tag, "value": count} for tag, count in liked_by_cust.items()]
    negative_key_words = [{"tagname": tag, "value": count} for tag, count in disliked_by_cust.items()]

    # Question
    netral_data = utils.get_netral_data(class_labels=class_labels, data=texts)
    questions_data = []
    if len(netral_data) > 0:
        is_questions, question_class_labels, question_predictions = utils.predict_question_batch(netral_data, model=question_model, tokenizer=tokenizer, preprocess=True)
        questions_data = utils.get_questions_or_assistance(netral_data, question_class_labels)

    q_usernames = utils.get_username(texts_to_index_map, questions_data, usernames)
    len_questions = len(questions_data)
    questions_usernames_map = [{"username": q_usernames[i], "text": questions_data[i]} for i in range(len_questions)]

    # Assistance
    is_assistances, assistance_class_labels, assistance_predictions = utils.predict_assistance_batch(texts, model=assistance_model, tokenizer=tokenizer, preprocess=True)
    assistances_data = utils.get_questions_or_assistance(texts, assistance_class_labels)

    # Return results in the new format
    return {
        "data": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "topstatus": {
                "positive": top_3_pos_comments_username,
                "negative": top_3_neg_comments_username
            },
            "key_words": {
                "positive": positive_key_words,
                "negative": negative_key_words
            },
            "questions": questions_usernames_map,
            "assistances": assistances_data
        }
    }