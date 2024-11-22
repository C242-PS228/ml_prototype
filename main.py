import gradio as gr
import re
import pandas as pd
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
from utils import emoji_dict, stop_words
from nltk.tokenize import RegexpTokenizer
import io

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = load_model("model/bert_attention_v4.h5")

# PREPROCESSING
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

# PREDICT
def predict_sentiment_batch(texts):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    tokenized_texts = tokenize_batch(preprocessed_texts)
    predictions = model.predict(tokenized_texts)
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    sentiments = [sentiment_labels[pred.argmax()] for pred in predictions]
    return sentiments

# OUTPUT
def summarize_sentiments(sentiments):
    summary = pd.Series(sentiments).value_counts().to_dict()
    return {
        "Positif": summary.get("Positif", 0),
        "Netral": summary.get("Netral", 0),
        "Negatif": summary.get("Negatif", 0),
    }

def create_csv_file(texts, sentiments):
    df = pd.DataFrame({"Input Text": texts, "Predicted Sentiment": sentiments})
    csv_path = "sentiment_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def create_excel_file(texts, sentiments):
    df = pd.DataFrame({"Input Text": texts, "Predicted Sentiment": sentiments})
    excel_path = "sentiment_results.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path

# GRADIO UI
with gr.Blocks() as sentiment_app:
    gr.Markdown("# Sentiment Classification App (Batch Mode)")
    with gr.Row():
        with gr.Column():
            input_texts = gr.TextArea(label="Enter Texts (One per Line)", placeholder="Enter multiple texts, each on a new line.")
            submit_btn = gr.Button("Classify Sentiments")
        with gr.Column():
            output_labels = gr.Dataframe(headers=["Input Text", "Predicted Sentiment"], interactive=False)
            summary_label = gr.Label(label="Summary of Sentiments")
            download_csv = gr.File(label="Download CSV")
            download_excel = gr.File(label="Download Excel")

    def process_texts(texts):
        texts_list = texts.split("\n")  
        sentiments = predict_sentiment_batch(texts_list)  
        summary = summarize_sentiments(sentiments)  
        summary_text = (
            f"Positif: {summary['Positif']} | "
            f"Netral: {summary['Netral']} | "
            f"Negatif: {summary['Negatif']}"
        )
        csv_path = create_csv_file(texts_list, sentiments)  
        excel_path = create_excel_file(texts_list, sentiments)  
        return pd.DataFrame({"Input Text": texts_list, "Predicted Sentiment": sentiments}), summary_text, csv_path, excel_path

    submit_btn.click(
        fn=process_texts,
        inputs=input_texts,
        outputs=[output_labels, summary_label, download_csv, download_excel],
    )

sentiment_app.launch(share=True)
