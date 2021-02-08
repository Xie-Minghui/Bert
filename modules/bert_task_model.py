# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/8 13:09
# @File    : bert_task_model.py

"""
file description:：

"""
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .bert import BertModel


class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights(config)
    
    def init_weights(self, config):
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()
    
    def forward(
            self,
            input_ids=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,  # 直接传入已经编码号的输入，而不是传入input_ids和position_ids等在进行Embedding
            output_attentions=None,
            output_hidden_states=None
            ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        output = {'loss': loss, 'logits': logits, 'hidden_states': outputs['hidden_states'],
                  'attention': outputs['attention']}
        
        return output
        