# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/8 9:43
# @File    : my_train.py

"""
file description:：

"""
import torch
import os
import sys

from data_load.tokenize_ import BertTokenizer
from modules.bert import BertModel
from utils.config import BertConfig
from utils.transfer import load_weights


if __name__ == "__main__":
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    text0 = '小明是谁？'
    text1 = '小明是一个学生'
    vocab_file = './bert-base-chinese/vocab.txt'
    # text = '你好呀'
    bert_tokenizer = BertTokenizer(vocab_file)
    output = bert_tokenizer.encode(text0, text1, add_special_tokens=True)  # max_length=10
    print(output.keys())
    print("input id:")
    print(output['input_ids'])
    print('token_type_ids:')
    print(output['token_type_ids'])
    print('attention_mask:')
    print(output['attention_mask'])
    
    config = BertConfig()
    bert = BertModel(config, add_pooling_layer=True)
    # token_type_ids
    model_state_dict = bert.state_dict()
    model_path = './bert-base-chinese/pytorch_model.bin'
    if os.path.exists(model_path):
        state_dict = torch.load_state_dict(model_path, map_location='gpu' if not torch.is_available() else 'cpu')
    else:
        print('Please download the model file')
        sys.exit()
        
    load_weights(bert, state_dict)
    print("nihao")
    