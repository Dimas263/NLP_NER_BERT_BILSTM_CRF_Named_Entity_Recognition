# NLP
# Named Entity Recognition (NER) - BERT - BILSTM -CRF


## <img src="https://img.icons8.com/external-smashingstocks-flat-smashing-stocks/64/000000/external-manager-hotel-smashingstocks-flat-smashing-stocks-2.png"/> **`Slamet Riyanto S.Kom., M.M.S.I.`**

## <img src="https://img.icons8.com/external-fauzidea-flat-fauzidea/64/undefined/external-man-avatar-avatar-fauzidea-flat-fauzidea.png"/> **`Dimas Dwi Putra`**

## Architecture
<img src="NER-BERT-BILSTM-CRF%20Architecture.png" width="9645">

## Dataset

B-I-O
```yaml
{0: 'O', 1: 'B', 2: 'I'}
```
Labels
```yaml
{'UNK': 0, 'plant': 1, 'disease': 2} {0: 'UNK', 1: 'plant', 2: 'disease'}
```
Json
```yaml
[
    {
      'id' : ...,
      'labels' : ['Truncate', 'Type Entity', 'Start Entity', 'End Entity', 'Entity Names'],
      'text': ...
    }
]
```
Example
```yaml
[
    {
      'id': 0,
      'labels': [['T0', 'plant', 46, 55, 'digitalis'],
       ['T1', 'disease', 64, 75, 'arrhythmias']],
      'text': 'studies on magnesium s mechanism of action in digitalis induced arrhythmias'
    },
    ...
]
```

## Eval
| Fine Tuning   | Biobert-Plant-Disease | Biobert-Plant-Disease | Biobert-Plant-Disease | Biobert-Plant-Disease |
| ------------- |-----------------------|-----------------------|-----------------------|-----------------------|
| Model         | Bert                  | Bert-CRF              | Bert-Bilstm           | Bert-Bilstm-CRF       |
| Batch Size    | 2                     | 2                     | 2                     | 2                     |
| Epoch         | 10                    | 10                    | 10                    | 10                    |
| Iterasi       | 393                   | 393                   | 393                   | 393                   |
| Step          | 3.930                 | 3.930                 | 3.930                 | 3.930                 |
| Learning Rate | 0,00003               | 0,00003               | 0,00003               | 0,00003               |
| Dropout       | 0,1                   | 0,1                   | 0,1                   | 0,1                   |
| Entitas       | (Plant)  (Disease)    | (Plant)  (Disease)    | (Plant)  (Disease)    | (Plant)  (Disease)    |
| Precision     | (0,86)   (0,66)       | (0,79)   (0,64)       | (0,87)   (0,68)       | (0,82)   (0,62)       |
| Recall        | (0,64)   (0,43)       | (0,64)   (0,41)       | (0,64)   (0,42)       | (0,64)   (0,44)       |
| F-1 Score     | (0,73)   (0,52)       | (0,71)   (0,5)        | (0,74)   (0,51)       | (0,72)   (0,51)       |
| Average/Total |                       |                       |                       |                       |  
| Precision     | 0,74                  | 0,71                  | 0,76                  | 0,7                   |
| Recall        | 0,51                  | 0,5                   | 0,51                  | 0,52                  |
| F-1 Score     | 0,61                  | 0,58                  | 0,61                  | 0,6                   |
| Eksekusi      | 0:22:35               | 1:01:40               | 0:25:28               | 1:04:07               |
| Device        | Cuda Tesla T4         | Cuda Tesla T4         | Cuda Tesla T4         | Cuda Tesla T4         |

## Predict
```yaml
effects of korean red ginseng extracts on neural tube defects and impairment of social interaction induced by prenatal exposure to valproic
{'plant': [['ginseng', 22]], 'disease': [['neural', 42], ['tube', 49], ['defects', 54]]}
```

## Output
### Save model output as [Pytorch .pt](output/) 

# **Other Content**

### **Websites Prediction**
#### [1. Django Websites Prediction For NER dan RE](https://github.com/Dimas263/Django-Websites_NER_RE)


### **Named Entity Recognition (NER)**
#### [1. NER Dataset Biomedical Plant-Disease Corpus](https://github.com/Dimas263/NLP_NER_Dataset_Biomedical_Plant-Disease_Corpus)
#### [2. NER CRF Named Entity Recognition](https://github.com/Dimas263/NLP_NER_CRF_Named_Entity_Recognition)
#### [3. NER BiLSTM Named Entity Recognition](https://github.com/Dimas263/NLP_NER_BILSTM_Named_Entity_Recognition)
#### [4. NER BERT Named Entity Recognition](https://github.com/Dimas263/NLP_NER_BERT_Named_Entity_Recognition)
#### [5. NER BiLSTM CRF Named Entity Recognition](https://github.com/Dimas263/NLP_NER_BILSTM_CRF_Named_Entity_Recognition)
#### [6. NER BERT BiLSTM CRF Named Entity Recognition](https://github.com/Dimas263/NLP_NER_BERT_BILSTM_CRF_Named_Entity_Recognition)


### **Relation Extraction (NER)**
#### [1. RE Dataset Biomedical Plant-Disease Corpus](https://github.com/Dimas263/NLP_RE_Dataset_Biomedical_Plant-Disease_Corpus)
#### [2. RE BERT Relation Extraction Biomedical](https://github.com/Dimas263/NLP_RE_BERT_Relation_Extraction_Biomedical)
#### [3. RE BiLSTM CRF Relation Extraction Biomedical](https://github.com/Dimas263/NLP_RE_BILSTM_CRF_Relation_Extraction_Biomedical)
