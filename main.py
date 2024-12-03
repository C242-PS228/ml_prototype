from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import utils  # File utils dengan helper functions

app = FastAPI()
model_url = "https://storage.googleapis.com/production_model_storage/perceivo_models/bert_attention_v11.h5"
question_model_url = "https://storage.googleapis.com/production_model_storage/perceivo_models/question_bert.h5"
assistance_model_url = "https://storage.googleapis.com/production_model_storage/perceivo_models/assistance_bert_v2.h5"

# Load the model and tokenizer
tokenizer = utils.load_tokenizer_from_folder("tokenizer")  # Load tokenizer from folder
model = utils.load_nlp_model_from_url(model_url)
question_model = utils.load_nlp_model_from_url(question_model_url)
assistance_model = utils.load_nlp_model_from_url(assistance_model_url)

nlp = utils.load_stanza_pipeline()  # Load NLP pipeline (e.g., Stanza)

class Comment(BaseModel):
    text: Optional[str] = Field(None, description="Comment text can be null or missing.")
    username: Optional[str] = Field(None, description="Username of the commenter.")

class RequestBody(BaseModel):
    comments: List[Comment]

@app.post("/predict")
async def predict_sentiments(data: RequestBody):
    texts = []
    usernames = []

    # Extract texts and usernames
    for comment in data.comments:
        if comment.text:
            texts.append(comment.text.strip())
            usernames.append(comment.username)

    if not texts:
        raise HTTPException(status_code=400, detail="All comments are null, empty, or missing text.")

    texts_to_index_map = {text: idx for idx, text in enumerate(texts)}
    preprocessed_texts = [utils.preprocess_text(text) for text in texts]

    # Predict sentiments
    sentiments, class_labels, predictions = utils.predict_sentiment_batch(preprocessed_texts, model=model, tokenizer=tokenizer, preprocess=False)

    # Summarize sentiments
    positive_count = sentiments.count("Positif")
    neutral_count = sentiments.count("Netral")
    negative_count = sentiments.count("Negatif")

    # Get top 3 positive and negative comments
    top_3_positive = utils.get_top_3_positive_comments(predictions, texts)
    top_3_negative = utils.get_top_3_negative_comments(predictions, texts)
    top_3_pos_username = utils.get_username(texts_to_index_map, top_3_positive, usernames)
    top_3_neg_username = utils.get_username(texts_to_index_map, top_3_negative, usernames)

    top_3_pos_comments_username = [{"username": top_3_pos_username[i], "text": top_3_positive[i]} for i in range(len(top_3_positive))]
    top_3_neg_comments_username = [{"username": top_3_neg_username[i], "text": top_3_negative[i]} for i in range(len(top_3_negative))]

    # Extract keywords
    liked_by_cust, disliked_by_cust = utils.get_key_words_and_clean_up(preprocessed_texts, class_labels, stanza=nlp, model=model, tokenizer=tokenizer)
    positive_key_words = [{"tagname": tag, "value": count} for tag, count in liked_by_cust.items()]
    negative_key_words = [{"tagname": tag, "value": count} for tag, count in disliked_by_cust.items()]

    # Questions
    netral_data = utils.get_netral_data(class_labels=class_labels, data=texts)
    questions_data = []
    if netral_data:
        is_questions, question_class_labels, question_predictions = utils.predict_question_batch(netral_data, model=question_model, tokenizer=tokenizer, preprocess=True)
        questions_data = utils.get_questions_or_assistance(netral_data, question_class_labels)

    q_usernames = utils.get_username(texts_to_index_map, questions_data, usernames)
    questions_usernames_map = [{"username": q_usernames[i], "text": questions_data[i]} for i in range(len(questions_data))]

    # Assistance
    is_assistances, assistance_class_labels, assistance_predictions = utils.predict_assistance_batch(texts, model=assistance_model, tokenizer=tokenizer, preprocess=True)
    assistances_data = utils.get_questions_or_assistance(texts, assistance_class_labels)

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
