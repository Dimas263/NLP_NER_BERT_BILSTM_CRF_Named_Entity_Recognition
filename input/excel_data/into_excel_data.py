import json

import pandas as pd
import xlsxwriter
import os
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('averaged_perceptron_tagger')

with open('./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/excel_data/ner_dataset.json', encoding='utf-8') as f:
    raw = json.load(f)

content_1 = []
content_2 = []
for i, item in enumerate(raw):
    text = item["text"]
    ids = item["id"]
    labels = item["labels"]

    labels_tags = []
    labels_word = []
    if len(labels) != 0:
        labels = [[label[0], label[2], label[3], label[4], label[1]] for label in labels]
        labels_word = [label[3] for label in labels]
        labels_tags = [label[4] for label in labels]

    texts = text
    lower_case = texts.lower()
    tokens = nltk.word_tokenize(lower_case)
    tags = nltk.pos_tag(tokens)
    counts = Counter(tag for word, tag in tags)

    tags_word = []
    tags_pos = []
    if len(tags) != 0:
        tags = [[pos_tag[0], pos_tag[1]] for pos_tag in tags]
        tags_word = [pos_tag[0] for pos_tag in tags]
        tags_pos = [pos_tag[1] for pos_tag in tags]

    content_1.append([labels_word, labels_tags])
    content_2.append([tags_word, tags_pos])

df_labels = pd.DataFrame(content_1, columns=['Word', 'Tag']).explode(['Word', 'Tag'])
print(df_labels)

print()

df_tags = pd.DataFrame(content_1, columns=['Word', 'Tag']).explode(['Word', 'Tag'])
print(df_tags)

print()

df_tags = pd.DataFrame(content_2, columns=['Word', 'POS']).explode(['Word', 'POS'])
print(df_tags)

print()

df_join = pd.merge(df_labels, df_tags, how='left', on=['Word'])
print(df_join)
