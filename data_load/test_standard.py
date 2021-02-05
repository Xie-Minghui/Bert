# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/2 19:13
# @File    : test_standard.py

"""
file description:：

"""

from transformers import BertTokenizer, BertForQuestionAnswering
import transformers
# from transformers import modeling_
import torch

MODEL_PATH = "../bert-base-chinese"
# 实例化tokenizer
tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese/vocab.txt")
# 导入bert的model_config
model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
# 首先新建bert_model
bert_model = transformers.BertModel.from_pretrained(MODEL_PATH,config = model_config)
# 最终有两个输出，初始位置和结束位置（下面有解释）
model_config.num_labels = 2
# 同样根据bert的model_config新建BertForQuestionAnswering
model = BertForQuestionAnswering(model_config)
model.bert = bert_model

model.eval()
question = "你好呀"
# 获取input_ids编码
input_ids = tokenizer.encode(question)
# 手动进行token_type_ids编码，可用encode_plus代替
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
print("input_ids:{}\ntoken_type_ids:{}".format(input_ids, token_type_ids))
# 得到评分,
# start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
# # 进行逆编码，得到原始的token
# all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

# print(all_tokens)