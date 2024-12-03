def get_key_words_and_clean_up(preprocessed_texts, class_labels, stanza, tokenizer, model):
    nlp = stanza
    pos_dict = {}
    neg_dict = {}

    for text, label in zip(preprocessed_texts, class_labels):
        previous_noun = None
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text

                if word_text in exclude_words:
                    previous_noun = None
                    continue
                
                if word.upos == "ADJ" or word_text in adjectives:
                    if previous_noun:
                        # Noun-adjective phrase
                        phrase = f"{previous_noun} {word_text}"
                        if label == 2:
                            pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                        elif label == 0:
                            neg_dict[phrase] = neg_dict.get(phrase, 0) + 1
                        previous_noun = None
                    else:
                        # Standalone adjective
                        if label == 2:
                            pos_dict[word_text] = pos_dict.get(word_text, 0) + 1
                        elif label == 0:
                            neg_dict[word_text] = neg_dict.get(word_text, 0) + 1

                # Handle nouns
                elif word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text

    pos_arr, neg_arr = get_tag_words(pos_dict, neg_dict)
    if len(pos_arr) > 0:
        pos_tokenized = tokenize_batch(pos_arr, tokenizer)
        true_pos_label = model.predict(pos_tokenized)
        class_labels_pos = np.argmax(true_pos_label, axis=1)

        for i, label in enumerate(class_labels_pos):
            if label != 2:
                word = pos_arr[i]
                del pos_dict[word]

    if len(neg_arr) > 0:
        neg_tokenized = tokenize_batch(neg_arr, tokenizer)
        true_neg_label = model.predict(neg_tokenized)
        class_labels_neg = np.argmax(true_neg_label, axis=1)

        for i, label in enumerate(class_labels_neg):
            if label != 0:
                word = neg_arr[i]
                del neg_dict[word]

    return pos_dict, neg_dict