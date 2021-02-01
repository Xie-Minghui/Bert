# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/1 16:38
# @File    : tokenizer.py

"""
file description:：

"""
import unicodedata


class BasicTokenizer(object):
    
    def __init__(self, do_lowercase=True, tokenize_chinese=False):
        self.do_lowercase = do_lowercase
        self.tokenize_chinese = tokenize_chinese
        pass
    
    def tokenize(self, text):
        text = self._clean_text(text)
        
        if self.tokenize_chinese:
            self._tokenize_chinese(text)
        
        tokens = self._whitespace_tokenize(text)
        return tokens
        
    
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return "".join(output)
    
    def _tokenize_chinese(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' {} '.format(char))
            else:
                output.append(char)
        
        return ''.join(output)
    
    def _whitespace_tokenize(self, text):
        text = text.strip()
        if not text:
            return []
        else:
            return text.split()
    
    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == ' ' or char == '\t' or char == '\n' or char == '\r':
            return True
        cat = unicodedata.category(char)
        if cat == 'Zs':
            return True
        
        return False
    
    def _is_control(self, char):
        """检查字符char是否是控制字符"""
        # 回车换行和tab理论上是控制字符，但是这里我们把它认为是whitespace而不是控制字符
        if char == '\t' or char == '\n' or char == '\r':
            return False
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            return True
        
        return False

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True
    
        return False
#  首先，转化为unicode，方便处理特殊字符

#  去除无意义字符

# 空格处理： 去除控制符号，将一些其他隔离符号（\t）转化为空格

#  判断是否是中文分词。其他分词直接使用空格分词；

#  中文分词：直接将每个词添加空格，分词。




