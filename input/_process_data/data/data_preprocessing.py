import re
import json
import pandas as pd
from pprint import pprint

df = pd.read_excel('./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/data/gold-standard-corpus.xlsx')

texts = []
for plant, disease, text in zip(df['plant'].tolist(), df['disease'].tolist(), df['sentence'].tolist()):
    text = text.replace(disease, f'<ENAMEX TYPE="disease">{disease}</ENAMEX>').replace(plant, f'<ENAMEX TYPE="plant">{plant}</ENAMEX>')
    texts.append(text)

df['text'] = texts

with open('./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/data/annotated-dataset-plant-disease-corpus.txt', 'w', encoding='utf-8') as f:
    for text in zip(df['text'].tolist()):
        f.write(f"{text}\n")

with open('./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/data/annotated-dataset-plant-disease-corpus.txt', 'r') as g:
    filedata = g.read()

filedata = filedata.replace("('", '"').replace(" ',)", '"')

with open('./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/data/annotated-dataset-plant-disease-corpus.txt', 'w') as g:
    g.write(filedata)

print("success converting gold-standard-corpus.xlsx to annotated-dataset-plant-disease-corpus.txt")
