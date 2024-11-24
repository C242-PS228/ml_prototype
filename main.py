import gradio as gr
import re
import pandas as pd
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
from utils import emoji_dict, stop_words, nouns, adjectives, exclude_words, get_top_3_positive_comments, get_top_3_negative_comments
from nltk.tokenize import RegexpTokenizer
import stanza
from collections import Counter
import io
import numpy as np

# Load models and tools
custom_dir = "./stanza_models"
nlp = stanza.Pipeline('id', dir=custom_dir)
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = load_model("model/bert_attention_v6.h5")

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

# PREDICT
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

# SENTIMENT ANALYSIS
exclude_words = ['gua', 'kak', 'gue', 'sih', 'kasih', 'banget', 'orang', 'bu', 'sumpah', 'gitu', 'bnyak', 'banyak', 'gt', 'gitu', 'duo', 'dua', 'satu', 'min', 'pesen', 'brp', 'berapa','memang', 'mmg', 'udh', 'udah', 'uda', 'niat', 'tp', 'tapi']

def analyze_sentiment(preprocessed_texts, class_labels):
    # preprocessed_texts = [preprocess_text(text) for text in texts]
    # class_labels = predict_sentiment_batch(preprocessed_texts)
    
    pos_counter = Counter()
    neg_counter = Counter()

    for text, label in zip(preprocessed_texts, class_labels):
        previous_noun = None
        doc = nlp(text)
        
        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text
                
                if word_text in exclude_words:
                    previous_noun = None
                    continue
                
                if word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text
                
                elif (word.upos == "ADJ" or word_text in adjectives) and previous_noun:
                    phrase = f"{previous_noun} {word_text}"
                    if label == 2:
                        pos_counter[phrase] += 1
                    elif label == 0:
                        neg_counter[phrase] += 1
                    previous_noun = None

    pos_common_words = pos_counter.most_common()
    neg_common_words = neg_counter.most_common()
    
    return pos_common_words, neg_common_words

def get_top_3_common_words(pos_common_words, neg_common_words):
    top_3_pos_words = [word for word, _ in pos_common_words[:3]]
    top_3_neg_words = [word for word, _ in neg_common_words[:3]]
    
    return top_3_pos_words, top_3_neg_words

def analyze_sentiment_top_3(texts, class_labels):
    pos_common_words, neg_common_words = analyze_sentiment(texts, class_labels)
    return get_top_3_common_words(pos_common_words, neg_common_words)

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
            output_labels = gr.Dataframe(headers=["Predicted Sentiment", "Input Text"], interactive=False)
            summary_label = gr.Label(label="Summary of Sentiments")
            top_3_com_pos = gr.Label(label="Yang disukai customer")
            top_3_com_neg = gr.Label(label="Yang tidak disukai customer")
            top_3_pos_comments_label = gr.Label(label="Top 3 Positive comments")
            top_3_neg_comments_label = gr.Label(label="Top 3 Negative comments")
            download_csv = gr.File(label="Download CSV")
            download_excel = gr.File(label="Download Excel")

    def process_texts(texts):
        texts_list = texts.split("\n")  
        preprocessed_texts = [preprocess_text(text) for text in texts_list]
        sentiments, class_labels, predictions = predict_sentiment_batch(preprocessed_texts, preprocess=False)  
        summary = summarize_sentiments(sentiments)  
        top_com_pos_words, top_com_neg_words = analyze_sentiment_top_3(preprocessed_texts, class_labels)
        top_3_pos_comments = get_top_3_positive_comments(predictions, texts_list)
        top_3_neg_comments = get_top_3_negative_comments(predictions, texts_list)
        
        # Convert top comments lists into strings
        top_3_pos_comments_text = "\n".join(top_3_pos_comments)  # Join the comments with newline
        top_3_neg_comments_text = "\n".join(top_3_neg_comments)  # Join the comments with newline
            
        summary_text = (
            f"Positif: {summary['Positif']} | "
            f"Netral: {summary['Netral']} | "
            f"Negatif: {summary['Negatif']}"
        )
        top_com_pos_text = ", ".join(top_com_pos_words)
        top_com_neg_text = ", ".join(top_com_neg_words)
        
        csv_path = create_csv_file(texts_list, sentiments)  
        excel_path = create_excel_file(texts_list, sentiments)  
        
        return pd.DataFrame({"Predicted Sentiment": sentiments, "Input Text": texts_list}), summary_text, top_com_pos_text, top_com_neg_text, top_3_pos_comments_text, top_3_neg_comments_text,csv_path, excel_path

    submit_btn.click(
        fn=process_texts,
        inputs=input_texts,
        outputs=[output_labels, summary_label, top_3_com_pos, top_3_com_neg, top_3_pos_comments_label, top_3_neg_comments_label,download_csv, download_excel],
    )

if __name__ == "__main__":
    sentiment_app.launch()
