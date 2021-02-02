# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/2 17:23
# @File    : test_tokenize.py

"""
file description:：

"""
from data_load.tokenize import BertTokenizer

if __name__ == "__main__":
    vocab_file = '../data/vocab.txt'
    text = '你好呀'
    bert_tokenizer = BertTokenizer(vocab_file)
    output = bert_tokenizer.encode(text, max_length=10, add_special_tokens=True)
    print(output)
