import gradio as gr
import re
import pandas as pd
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
from utils import emoji_dict, stop_words
from nltk.tokenize import RegexpTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = load_model("model/bert_attention.h5")

# Helper functions for text preprocessing
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
    # Preprocess all texts at once
    preprocessed_texts = [preprocess_text(text) for text in texts]
    # Tokenize all texts as a batch
    tokenized_texts = tokenize_batch(preprocessed_texts)
    # Predict sentiment for the entire batch
    predictions = model.predict(tokenized_texts)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    # Map predictions to sentiment labels
    sentiments = [sentiment_labels[pred.argmax()] for pred in predictions]
    return sentiments

# Gradio interface
with gr.Blocks() as sentiment_app:
    gr.Markdown("# Sentiment Classification App (Batch Mode)")
    with gr.Row():
        with gr.Column():
            input_texts = gr.TextArea(label="Enter Texts (One per Line)", placeholder="Enter multiple texts, each on a new line.")
            submit_btn = gr.Button("Classify Sentiments")
        with gr.Column():
            output_labels = gr.Dataframe(headers=["Input Text", "Predicted Sentiment"], interactive=False)

    # Click event for batch prediction
    def process_texts(texts):
        texts_list = texts.split("\n")  # Split the textarea input into individual lines
        sentiments = predict_sentiment_batch(texts_list)  # Predict all at once
        return pd.DataFrame({"Input Text": texts_list, "Predicted Sentiment": sentiments})
    
    submit_btn.click(fn=process_texts, inputs=input_texts, outputs=output_labels)

# Launch the app
sentiment_app.launch()
