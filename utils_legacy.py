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


def get_key_words_and_clean_up(texts, class_labels, stanza, tokenizer, model, preprocess=False):
    if preprocess:
        texts = [preprocess_text_delete_emoji(text) for text in texts]
        
    nlp = stanza
    pos_dict = {}
    neg_dict = {}

    for text, label in zip(texts, class_labels):
        previous_noun = None
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text

                if word_text in exclude_words:
                    previous_noun = None
                    continue
                # Handle nouns
                if (word.upos == "NOUN" or word_text in nouns) and word_text not in exclude_nouns or (word_text in negations):
                    previous_noun = word_text
                
                elif (word.upos == "ADJ" or word.upos == 'NOUN') or word_text in adjectives and previous_noun:
                    # Noun-adjective phrase
                    phrase = f"{previous_noun} {word_text}"
                    if label == 2:
                        pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                    elif label == 0:
                        neg_dict[phrase] = neg_dict.get(phrase, 0) + 1
                    previous_noun = None


    pos_arr, neg_arr = get_array_words(pos_dict, neg_dict)
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


def get_key_words_and_clean_up_v2(preprocessed_texts, class_labels, stanza, tokenizer, model):
    nlp = stanza
    pos_dict = {}
    neg_dict = {}


    for text, label in zip(preprocessed_texts, class_labels):
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                # Filter based on dependency and POS
                if word.deprel == "amod" and len(word.text) > 2:
                    head_word = sent.words[word.head - 1]
                    if head_word.upos == "NOUN" and len(head_word.text) > 2:
                        noun = head_word.text.lower()
                        adj = word.text.lower()

                        # Skip if either adjective or noun is in custom stopwords
                        if noun in exclude_nouns or adj in exclude_nouns:
                            continue

                        # Form the phrase and count
                        phrase = f"{noun} {adj}"
                        if label == 2:  # Positive
                            pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                        elif label == 0:  # Negative
                            neg_dict[phrase] = neg_dict.get(phrase, 0) + 1

    # Apply frequency threshold filter: keep phrases appearing more than once
    pos_dict = {k: v for k, v in pos_dict.items() if v > 1}
    neg_dict = {k: v for k, v in neg_dict.items() if v > 1}

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

def get_key_words(preprocessed_texts, class_labels, stanza):
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

                # Handle nouns
                if word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text
                     # Standalone nouns
                    if label == 2:
                        pos_dict[word_text] = pos_dict.get(word_text, 0) + 1
                    elif label == 0:
                        neg_dict[word_text] = neg_dict.get(word_text, 0) + 1

                # Handle adjectives with or without nouns
                elif word.upos == "ADJ" or word_text in adjectives:
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

    return pos_dict, neg_dict


def get_key_words_and_clean_up(texts, class_labels, stanza, tokenizer, model, preprocess=False):
    if preprocess:
        texts = [preprocess_text_delete_emoji_and_normalize(text) for text in texts]
    print(texts)
    nlp = stanza
    pos_dict = {}
    neg_dict = {}

    for text, label in zip(texts, class_labels):
        previous_noun = None
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text
                if len(word_text) == 1:
                    continue 

                word_text = re.sub(r'ny[ae]*$', '', word_text, flags=re.IGNORECASE)

                if word_text in exclude_words or word_text in exclude_nouns:
                    previous_noun = None
                    continue
                
                if word.upos == "ADJ" or word_text in adjectives:
                    if previous_noun:
                        # print(word_text)
                        # Noun-adjective phrase
                        phrase = f"{previous_noun} {word_text}"
                        print(phrase)
                        if label == 2:
                            pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                        elif label == 0:
                            neg_dict[phrase] = neg_dict.get(phrase, 0) + 1
                        previous_noun = None

                # Handle nouns
                elif word.upos == "NOUN" or word_text in nouns or word_text in negations:
                    # print(word_text)

                    previous_noun = word_text
                    
    # print(f"pp {neg_dict}")
    pos_arr, neg_arr = get_array_words(pos_dict, neg_dict)
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

def get_key_words_and_clean_up(texts, class_labels, stanza, tokenizer, model, preprocess=False):
    if preprocess:
        texts = [preprocess_text_delete_emoji_and_normalize(text) for text in texts]
    nlp = stanza
    pos_dict = {}
    neg_dict = {}
    pos_arr = []
    neg_arr = []

    for text, label in zip(texts, class_labels):
        previous_noun = None
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text
                if len(word_text) == 1:
                    continue 

                word_text = re.sub(r'ny[ae]*$', '', word_text, flags=re.IGNORECASE)

                if word_text in exclude_words or word_text in exclude_nouns:
                    previous_noun = None
                    continue
                
                if word.upos == "ADJ" or word_text in adjectives:
                    if previous_noun:
                        # print(word_text)
                        # Noun-adjective phrase
                        phrase = f"{previous_noun} {word_text}"
                        print(phrase)
                        if label == 2:
                            pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                            pos_arr.append(phrase)
                        elif label == 0:
                            neg_dict[phrase] = neg_dict.get(phrase, 0) + 1
                            neg_arr.append(phrase)
                        previous_noun = None

                # Handle nouns
                elif word.upos == "NOUN" or word_text in nouns or word_text in negations:
                    # print(word_text)

                    previous_noun = word_text
                    
    # print(f"pp {neg_dict}")
    # pos_arr, neg_arr = get_array_words(pos_dict, neg_dict)
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

    return pos_arr, neg_arr