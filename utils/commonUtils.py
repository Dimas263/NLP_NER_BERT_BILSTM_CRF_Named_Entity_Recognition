# coding=utf-8
import functools
import random
import os
import json
import logging
import time
import pickle
import numpy as np
import torch


def timer(func):
    """
    function timer
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("{}total time{:.4f} second".format(func.__name__, end - start))
        return res

    return wrapper


def set_seed(seed=123):
    """
    Set random number seeds to ensure reproducible experiments
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    configure log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_json(data_dir, data, desc):
    """save as json"""
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(data_dir, desc):
    """read as json"""
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_pkl(data_dir, data, desc):
    """save .pkl"""
    with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'wb') as f:
        pickle.dump(data, f)


def read_pkl(data_dir, desc):
    """read .pkl"""
    with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'rb') as f:
        data = pickle.load(f)
    return data


def fine_grade_tokenize(raw_text, tokenizer):
    """
    The sequence tagging task BERT tokenizer may lead to tag offset,
    tokenize with char-level
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens
