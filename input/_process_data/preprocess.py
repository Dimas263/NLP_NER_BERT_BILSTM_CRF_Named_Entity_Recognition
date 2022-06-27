import os
import sys
import re

# from helper import remove_hex, remove_multiple_whitespace
import helper


def convert_raw_to_enimex(input_file: str, output_file: str):
    with open(input_file, 'r', encoding="ascii", errors='ignore') as f, open(output_file, 'w') as out:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i][:-1]
            while i < len(lines) and line[-1] != "\"":
                i += 1
                if i < len(lines):
                    line += lines[i][:-1]
            i += 1
            line = line[1:-1]
            line = line.replace("<", " <").replace(">", "> ")
            line = line.replace("\n", "")
            line = line.replace("	", "")
            line = helper.remove_multiple_whitespace(line)
            line = helper.remove_hex(line)
            line = helper.remove_space_between_quantity(line)
            line = "\"" + line + "\""
            line += '\n'
            out.write(line)


def convert_enimex_to_stanford(input_file: str, output_file: str):
    '''
    Convert ENAMEX Named-Entity annotated file to Stanford NLP format (token-based)
    @Author research.dimas@gmail
    ENAMEX example:
    Studies on magnesium\'s mechanism of action in <ENAMEX TYPE="plant">digitalis</ENAMEX> -induced <ENAMEX TYPE="disease">arrhythmias</ENAMEX> .
    '''

    START_PATTERN = re.compile(r'^(.*?)<ENAMEX$', re.I)
    END_SINGLE_PATTERN = re.compile(r'^TYPE="(.*?)">(.*?)</ENAMEX>(.*?)$', re.I)
    TYPE_PATTERN = re.compile(r'^TYPE="(.*?)">(.*?)$', re.I)
    END_MULTI_PATTERN = re.compile(r'^(.*?)</ENAMEX>(.*?)$', re.I)
    EOS_PATTERN = re.compile(r'^([^<>]*)\.?	(\d+)$', re.I)
    NON_ENTITY_TYPE = 'O'

    def check_and_process_eos(token):
        match = re.match(EOS_PATTERN, token)
        if match:
            out.write(match.group(1) + '	' + cur_type + '\n')
            out.write('.' + '	' + cur_type + '\n')
            out.write('\n')
            return True
        return False

    cur_type = NON_ENTITY_TYPE
    # print(infile)
    with open(input_file, 'r', encoding="ascii", errors='ignore') as f, open(output_file, 'w') as out:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i][:-1]
            i += 1
            line = helper.remove_multiple_whitespace(line)
            for token in line.strip().split(' '):
                token = token.strip()
                if not token:
                    continue

                match = re.match(START_PATTERN, token)
                if match:
                    if match.group(1):
                        out.write(match.group(1) + '	' +
                                  NON_ENTITY_TYPE + '\n')
                    continue

                match = re.match(END_SINGLE_PATTERN, token)
                if match:
                    out.write(match.group(2) + '	' + match.group(1) + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(match.group(3)):
                        out.write(match.group(3) + '	' + cur_type + '\n')
                    continue

                match = re.match(TYPE_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    out.write(match.group(2) + '	' + cur_type + '\n')
                    continue

                match = re.match(END_MULTI_PATTERN, token)
                if match:
                    out.write(match.group(1) + '	' + cur_type + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(match.group(2)):
                        out.write(match.group(2) + '	' + cur_type + '\n')
                    continue

                if check_and_process_eos(token):
                    continue

                out.write(token + '	' + cur_type + '\n')


def convert_stanford_to_bio(input_file: str, output_file: str):
    '''
    Convert ENAMEX Named-Entity annotated file to Stanford NLP format (token-based)
    @Author research.dimas@gmail
    ENAMEX example (2 sentences):
    Studies on magnesium\'s mechanism of action in <ENAMEX TYPE="plant">digitalis</ENAMEX> -induced <ENAMEX TYPE="disease">arrhythmias</ENAMEX> .
    '''

    NON_ENTITY_TYPE = 'O'

    cur_type = NON_ENTITY_TYPE
    with open(input_file, 'r', encoding="ascii", errors='ignore') as f, open(output_file, 'w') as out:
        prev = None
        prev_dot = False  # avoid printing double dot
        is_last = False
        for line in f.readlines():
            tokens = line.split('	')
            token, cur_type = tokens[0], tokens[1][:-1]
            if not token or token == "":
                continue

            if len(token) > 2 and token[0] == "\"" and token[-1] == "\"":
                token = token[1:-1]
            elif len(token) > 1 and token[0] == "\"":
                token = token[1:]
                out.write('\n')
            elif len(token) > 1 and token[-1] == "\"":
                token = token[:-1]

            if token == "\"":
                if not prev_dot:
                    out.write("." + '	' + NON_ENTITY_TYPE + '\n')
                    prev_dot = True
                    out.write('\n')
                prev = None
            else:
                token = token.lower()
                if token[-1] == "\"":
                    token = token[:-1]
                    is_last = True

                if cur_type == NON_ENTITY_TYPE:
                    out.write(token + '	' + cur_type + '\n')
                else:
                    if not prev:
                        out.write(token + '	B-' + cur_type + '\n')
                    else:
                        if prev == cur_type:
                            out.write(token + '	I-' + cur_type + '\n')
                        else:
                            out.write(token + '	B-' + cur_type + '\n')
                prev = cur_type
                prev_dot = False

                if is_last:
                    prev = None
                    if not prev_dot:
                        out.write("." + '	' + NON_ENTITY_TYPE + '\n')
                        prev_dot = True
                        out.write('\n')
                    is_last = False


def filter_bio(input_file: str, output_file: str):
    def filter(s):
        res = []
        for token in s[:-1]:  # unfilter last sentence
            word = token.split("	")[0]
            tag = token.split("	")[1]
            word = helper.remove_punctuation(word)
            word = helper.remove_multiple_whitespace(word)

            if word != "":
                res.append(word + "	" + tag)
        return "".join(res)

    with open(input_file, 'r', encoding="ascii", errors='ignore') as f, open(output_file, 'w') as out:
        l = 0
        s = []
        for line in f.readlines():
            if line[:-1] == "":
                if l > 3:
                    s = filter(s)
                    out.write(s)
                    out.write("\n")
                l = 0
                s = []
            else:
                l += 1
                s.append(line)


if __name__ == "__main__":
    input_file = "./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/data/annotated-dataset-plant-disease-corpus.txt"
    output_file = "./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/BIO/enimex.txt"

    convert_raw_to_enimex(input_file=input_file, output_file=output_file)

    input_file = output_file
    output_file = "./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/BIO/stanford.txt"

    convert_enimex_to_stanford(input_file=input_file, output_file=output_file)

    input_file = output_file
    output_file = "./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/BIO/BIO.txt"

    convert_stanford_to_bio(input_file=input_file, output_file=output_file)

    input_file = output_file
    output_file = "./drive/MyDrive/bert_bilstm_crf_named_entity_recognition/BERT-BILSTM-CRF-NER/input/_process_data/BIO/final-data.txt"

    filter_bio(input_file=input_file, output_file=output_file)
