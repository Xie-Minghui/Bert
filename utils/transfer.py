# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/8 11:15
# @File    : transfer.py

"""
file description:：

"""


def load_weights(model, state_dict):
    old_keys = []
    new_keys = []
    
    for key in state_dict.keys():
        if "gamma" in key:
            new_key = key.replace("gamma", 'weight') # weight
            new_keys.append(new_key)
            old_keys.append(key)
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')  # bias
            new_keys.append(new_key)
            old_keys.append(key)
    
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    
    model_state_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            pretrained_dict[k] = v
    
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
        
    