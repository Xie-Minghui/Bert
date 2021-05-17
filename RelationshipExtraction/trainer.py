# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/8 14:28
# @File    : trainer.py

"""
file description:：

"""
import numpy as np

import torch.optim as optim

import torch.nn.functional
from torch.utils.data import Dataset, DataLoader

import torch

import argparse

# from transformers import BertForSequenceClassification
# from transformers import BertConfig
from modules.bert_task_model import BertForSequenceClassification
from utils.config import BertConfig
from utils.transfer import load_weights


from transformers import BertModel
from tqdm import trange

from RelationshipExtraction.loader import load_train
from RelationshipExtraction.loader import load_dev

from RelationshipExtraction.loader import map_id_rel
import random
import os
import sys


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(44)

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

USE_CUDA = torch.cuda.is_available()
# USE_CUDA=False

data = load_train()
train_text = data['text']
train_mask = data['mask']
train_label = data['label']

train_text = [t.numpy() for t in train_text]
train_mask = [t.numpy() for t in train_mask]

train_text = torch.tensor(train_text)
train_mask = torch.tensor(train_mask)
train_label = torch.tensor(train_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

data = load_dev()
dev_text = data['text']
dev_mask = data['mask']
dev_label = data['label']

dev_text = [t.numpy() for t in dev_text]
dev_mask = [t.numpy() for t in dev_mask]

dev_text = torch.tensor(dev_text)
dev_mask = torch.tensor(dev_mask)
dev_label = torch.tensor(dev_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

print("--eval data--")
print(dev_text.shape)
print(dev_mask.shape)
print(dev_label.shape)

# exit()
# USE_CUDA=False

if USE_CUDA:
    print("using GPU")

print('now1')
train_dataset = torch.utils.data.TensorDataset(train_text, train_mask, train_label)
dev_dataset = torch.utils.data.TensorDataset(dev_text, dev_mask, dev_label)


def get_train_args():
    labels_num = len(rel2id)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--nepoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_labels', type=int, default=labels_num)
    parser.add_argument('--data_path', type=str, default='.')
    opt = parser.parse_args()
    print(opt)
    return opt


def get_model(opt):
    # model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=opt.num_labels)
    config = BertConfig()
    bert_cls = BertForSequenceClassification(config)
    model_path = '../bert-base-chinese/pytorch_model.bin'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)  # , map_location='gpu' if torch.cuda.is_available() else 'cpu'
    else:
        print('Please download the model file')
        sys.exit()
    load_weights(bert_cls, state_dict)
    return bert_cls


def eval(net, dataset, batch_size):
    net.eval()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        iter = 0
        for text, mask, y in train_iter:
            iter += 1
            if text.size(0) != batch_size:
                break
            text = text.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)
            
            if USE_CUDA:
                text = text.cuda()
                mask = mask.cuda()
                y = y.cuda()
            
            outputs = net(text, attention_mask=mask, labels=y)
            # print(y)
            loss, logits = outputs['loss'], outputs['logits']
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
            s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))
        acc = (1.0 * correct.numpy()) / total
        print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total, "Acc:", acc)
        return acc


def train(net, dataset, num_epochs, learning_rate, batch_size):
    net.train()
    print('nihao')
    print(num_epochs)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0)
    # optimizer = AdamW(net.parameters(), lr=learning_rate)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    pre = 0
    for epoch in trange(num_epochs):
        print(epoch)
        correct = 0
        total = 0
        iter = 0
        for text, mask, y in train_iter:
            iter += 1
            optimizer.zero_grad()
            # print(type(y))
            # print(y)
            if text.size(0) != batch_size:
                break
            text = text.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)
            if USE_CUDA:
                text = text.cuda()
                mask = mask.cuda()
                y = y.cuda()
            # print(text.shape)
            # loss, logits = net(text, attention_mask=mask,labels=y)
            outputs = net(text, attention_mask=mask, labels=y)
            loss, logits = outputs['loss'], outputs['logits']
            # print(type(tmp))
            # print(tmp)

            # print(y)
            # print(loss.shape)
            # print("predicted",predicted)
            # print("answer", y)
            loss.backward()
            optimizer.step()
            # print(outputs[1].shape)
            # print(output)
            # print(outputs[1])
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
        loss = loss.detach().cpu()
        print("epoch ", str(epoch), " loss: ", loss.mean().numpy().tolist(), "right", correct.cpu().numpy().tolist(),
              "total", total, "Acc:", correct.cpu().numpy().tolist() / total)
        acc = eval(model, dev_dataset, 32)
        if acc > pre:
            pre = acc
            torch.save(model, str(acc) + 'model.pth')
            print('Save Model....')
    return


opt = get_train_args()
model = get_model(opt)
# model=nn.DataParallel(model,device_ids=[0,1])
if USE_CUDA:
    model = model.cuda()

# eval(model,dev_dataset,8)

train(model, train_dataset, 16, 0.002, 4)
eval(model,dev_dataset,8)