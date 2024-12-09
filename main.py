import gradio as gr
import pandas as pd
import utils

# Load models and tools
nlp = utils.load_stanza_pipeline()
tokenizer = utils.load_tokenizer('tokenizer')
model = utils.load_nlp_model("model/bert_attention_v12.h5")
question_model = utils.load_nlp_model("model/question_bert.h5")
assistance_model = utils.load_nlp_model("model/assistance_bert.h5")
gen_ai_model = utils.load_vertex_model()

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
            customer_questions = gr.Label(label="Pertanyaan customer")
            seek_assistance = gr.Label(label="Meminta bantuan")
            resume_label = gr.Label(label="Resume")
            download_csv = gr.File(label="Download CSV")
            download_excel = gr.File(label="Download Excel")

    def process_texts(texts):
        texts_list = texts.split("\n")  
        preprocessed_texts = [utils.preprocess_text_and_normalize(text) for text in texts_list]
        sentiments, class_labels, predictions = utils.predict_sentiment_batch(preprocessed_texts, model=model, tokenizer=tokenizer, preprocess=False)  
        summary = summarize_sentiments(sentiments)  

        # Key Words
        filtered_comments_keywords, filtered_class_labels = utils.limit_and_filter_comments_400(texts_list, class_labels=class_labels)
        # print(f"woii {len(filtered_comments_keywords)}")
        # print(f"woii {len(filtered_class_labels)}")
        top_com_pos_dict, top_com_neg_dict = utils.get_key_words_and_clean_up(filtered_comments_keywords, filtered_class_labels, stanza=nlp, tokenizer=tokenizer, model=model, preprocess=True)
        top_com_pos_words, top_com_neg_words = [word for word, _ in top_com_pos_dict.items()], [word for word, _ in top_com_neg_dict.items()]
        # top_com_pos_words, top_com_neg_words = [], []

        # Top Positive and Top Negative
        top_3_pos_comments = utils.get_top_3_positive_comments(predictions, texts_list)
        top_3_neg_comments = utils.get_top_3_negative_comments(predictions, texts_list)
        
        # Convert top comments lists into strings
        top_3_pos_comments_text = "\n".join(top_3_pos_comments)  # Join the comments with newline
        top_3_neg_comments_text = "\n".join(top_3_neg_comments)  # Join the comments with newline

        # Question
        netral_data = utils.get_netral_data(class_labels=class_labels, data=texts_list)
        questions_str = ''
        seek_assistance_str = ''
        if len(netral_data) > 0:
            is_questions, question_class_labels, question_predictions = utils.predict_question_batch(netral_data, model=question_model, tokenizer=tokenizer, preprocess=True)
            questions_data = utils.get_questions_or_assistance(netral_data, question_class_labels)
            # print(question_class_labels)
            # print(is_questions)
            questions_str = '| '.join(questions_data)

        is_assistances, assistance_class_labels, assistance_predictions = utils.predict_assistance_batch(preprocessed_texts, model=assistance_model, tokenizer=tokenizer)
        # print(f"woiii {assistance_predictions}")
        # print(f"ppp {assistance_class_labels}")
        # print(f"akwowkii {is_assistances}")
        # print(f"texts {texts}")
        # print(f"netral_data {netral_data}")
        assistances_data = utils.get_questions_or_assistance(texts_list, assistance_class_labels)
        assistances_str = '| '.join(assistances_data)

        summary_text = (
            f"Positif: {summary['Positif']} | "
            f"Netral: {summary['Netral']} | "
            f"Negatif: {summary['Negatif']}"
        )
        top_com_pos_text = ", ".join(top_com_pos_words)
        top_com_neg_text = ", ".join(top_com_neg_words)
        
        csv_path = create_csv_file(texts_list, sentiments)  
        excel_path = create_excel_file(texts_list, sentiments)  

        # Vertex ai
        gen_ai_input = utils.create_gen_ai_input(texts_list)
        print(gen_ai_input)
        resume_generated = utils.generate_resume(gen_ai_input, model=gen_ai_model)

        
        return pd.DataFrame({"Predicted Sentiment": sentiments, "Input Text": texts_list}), summary_text, top_com_pos_text, top_com_neg_text, top_3_pos_comments_text, top_3_neg_comments_text, questions_str, assistances_str, resume_generated,csv_path, excel_path

    submit_btn.click(
        fn=process_texts,
        inputs=input_texts,
        outputs=[output_labels, summary_label, top_3_com_pos, top_3_com_neg, top_3_pos_comments_label, top_3_neg_comments_label, customer_questions, seek_assistance, resume_label, download_csv, download_excel],
    )

if __name__ == "__main__":
    sentiment_app.launch()
