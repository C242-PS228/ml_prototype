import gradio as gr
import pandas as pd
import utils

# Load models and tools
nlp = utils.load_stanza_pipeline()
tokenizer = utils.load_tokenizer('tokenizer')
model = utils.load_nlp_model("model/bert_attention_v6.h5")

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
        preprocessed_texts = [utils.preprocess_text(text) for text in texts_list]
        sentiments, class_labels, predictions = utils.predict_sentiment_batch(preprocessed_texts, model=model, tokenizer=tokenizer, preprocess=False)  
        summary = summarize_sentiments(sentiments)  
        top_com_pos_words, top_com_neg_words = utils.analyze_sentiment_top_3(preprocessed_texts, class_labels, stanza=nlp)
        top_3_pos_comments = utils.get_top_3_positive_comments(predictions, texts_list)
        top_3_neg_comments = utils.get_top_3_negative_comments(predictions, texts_list)
        
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
