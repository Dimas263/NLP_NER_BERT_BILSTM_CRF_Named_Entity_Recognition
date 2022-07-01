# NLP - BERT-BILSTM-CRF Named-Entity_Recognition

# [Architecture](https://dimas263.github.io/work/data/NER%20Architecture.html)

<img src="Architecture-NER-BERT-BILSTM-CRF.png">

# Json Dataset

```yaml
[
{
  'id' : ...,
  'labels' : ['Truncate', 'Type Entity', 'Start Entity', 'End Entity', 'Entity Names'],
  'text': ...
}
]
```

```yaml
[
{'id': 0,
  'labels': [['T0', 'plant', 46, 55, 'digitalis'],
   ['T1', 'disease', 64, 75, 'arrhythmias']],
  'text': 'studies on magnesium s mechanism of action in digitalis induced arrhythmias'},
 {'id': 1,
  'labels': [['T0', 'plant', 41, 50, 'digitalis'],
   ['T1', 'disease', 59, 70, 'arrhythmias']],
  'text': 'the mechanism by which magnesium affects digitalis induced arrhythmias was studied in dogs with and without beta receptor'},
 {'id': 2,
  'labels': [['T0', 'plant', 147, 156, 'digitalis'],
   ['T1', 'disease', 163, 174, 'arrhythmias']],
  'text': 'magnesium s direct effect on calcium and potassium fluxes across the myocardial cell membrane may be the mechanism of its antiarrhythmic action in digitalis toxic arrhythmias'},
 {'id': 3,
  'labels': [['T0', 'plant', 21, 26, 'green'],
   ['T1', 'plant', 27, 30, 'tea'],
   ['T2', 'disease', 60, 64, 'skin'],
   ['T3', 'disease', 65, 75, 'papillomas']],
  'text': 'inhibitory effect of green tea on the growth of established skin papillomas in'},
  ...
]
```
# Labels
```yaml
BIO = {0: 'O', 1: 'B', 2: 'I'}
Labels = {'UNK': 0, 'plant': 1, 'disease': 2} {0: 'UNK', 1: 'plant', 2: 'disease'}
```

# Config

```javascript
! python main.py \
--bert_dir=model/biobert-plant-disease/ \
--data_dir=input/ \
--log_dir=logs/ \
--output_dir=output/checkpoints/ \
--bio_tags=3 \
--att_tags=3 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=2 \
--train_epochs=10 \
--eval_batch_size=2 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--lstm_hidden=128 \
--num_layers=1 \
--use_lstm='True' \
--use_crf='True' \
--dropout_prob=0.1 \
--dropout=0.1
```

# Scores

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

# Predict

```yaml
effects of korean red ginseng extracts on neural tube defects and impairment of social interaction induced by prenatal exposure to valproic
{'plant': [['ginseng', 22]], 'disease': [['neural', 42], ['tube', 49], ['defects', 54]]}
```
# Model Output
```javascript
-rwxrwxrwx 1 dimas dimas 417M Jun 26 12:58 bert_bilstm_crf/model.pt
-rwxrwxrwx 1 dimas dimas 417M Jun 26 13:35 bert_bilstm/model.pt
-rwxrwxrwx 1 dimas dimas 415M Jun 26 14:23 bert_crf/model.pt
-rwxrwxrwx 1 dimas dimas 415M Jun 26 14:54 bert/model.pt
```