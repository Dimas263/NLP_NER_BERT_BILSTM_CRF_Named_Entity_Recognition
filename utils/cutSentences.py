import re


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # single character break
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # English
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # Chinese
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sent_for_bert(text, max_seq_len):
    # Split sentences, fine-grained clauses, and then re-merge
    sentences = []

    sentences_v1 = cut_sentences_v1(text)
    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)

    assert ''.join(sentences) == text

    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]

        end_index_ = start_index_ + 1
        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_

        merged_sentences.append(tmp_text)

    return merged_sentences


def refactor_labels(sent, labels, start_index):
    """
    After the clause, the offset of labels needs to be reconstructed
    :param sent: the split and rejoined sentence
    :param labels: original document-level labels
    :param start_index: the starting offset of the sentence in the document
    :return (type, entity, offset)
    """
    new_labels = []
    end_index = start_index + len(sent)
    # _label: TI, entity type, entity start position, entity end position, entity name)
    for _label in labels:
        if start_index <= _label[2] <= _label[3] <= end_index:
            new_offset = _label[2] - start_index

            assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

            new_labels.append((_label[1], _label[-1], new_offset))
        # label truncated case
        elif _label[2] < end_index < _label[3]:
            raise RuntimeError(f'{sent}, {_label}')

    return new_labels


if __name__ == '__main__':
    raw_examples = [{
        "text": "conclusions tea and coffee drinking may decrease the risk of oral cavity cancer through antioxidant components which play a role in the repair of cellular",
        "labels": [
            ["T0", "plant", 12, 15, "tea"],
            ["T1", "disease", 61, 65, "oral"],
            ["T2", "disease", 66, 72, "cavity"],
            ["T3", "disease", 73, 79, "cancer"]
        ]
    }]
    for i, item in enumerate(raw_examples):
        text = item['text']
        print(text[:90])
        sentences = cut_sent_for_bert(text, 90)
        start_index = 0

        for sent in sentences:
            labels = refactor_labels(sent, item['labels'], start_index)
            start_index += len(sent)

            print(sent)
            print(labels)
