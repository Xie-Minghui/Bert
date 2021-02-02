# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/2/1 16:38
# @File    : tokenizer.py

"""
file description:：

"""
import unicodedata
import os
import collections

from .tokenize_utils import (
    PaddingStrategy,
    TruncationStrategy
)

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    else:
        return text.split()

def load_vocab(vocab_file):
    tokens2ids = collections.OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            line = line.rstrip('\n')
            tokens2ids[line] = cnt
            cnt += 1
    return tokens2ids
        

class BertTokenizer():
    def __init__(self,
                 vocab_file,
                 do_lowercase=True,
                 do_basic_tokenize=True,
                 tokenize_chinese=False,
                 never_split=None,
                 unk_token='[UNK]',
                 sep_token='[SEP]',
                 pad_token='[PAD]',
                 cls_token='[CLS]',
                 mask_token='[MASK]'
                 ):
    
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.tokens2ids = load_vocab(vocab_file)
        self.ids2tokens = collections.OrderedDict([(id, token) for token, id in self.tokens2ids.items()])
        self.do_lowercase = do_lowercase
        self.tokenize_chinese = tokenize_chinese
        self.never_split = never_split
        self.do_basic_tokenize = do_basic_tokenize
        
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                                        do_lowercase=do_lowercase,
                                        tokenize_chinese=tokenize_chinese,
                                        never_split=never_split
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.tokens2ids, unk_token=unk_token)
    
    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                if token in self.never_split:
                    split_tokens.append(token)  # 添加一个新元素
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)  # 延续最后一个元素
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        
        return split_tokens
    
    def encode(self,
               text,
               text_pair=None,
               padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
               truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
               ):
    
    def _convert_token2id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokens2ids.get(token, self.tokens2ids.get(self.unk_token))

class BasicTokenizer():  # 没必要继承object类，因为python3默认继承object类，里面有很多高级特性
    
    def __init__(self, do_lowercase=True, tokenize_chinese=False, never_split=None):
        self.do_lowercase = do_lowercase
        self.tokenize_chinese = tokenize_chinese
        self.never_split = never_split
        
    def tokenize(self, text):
        text = self._clean_text(text)
        
        if self.tokenize_chinese:
            self._tokenize_chinese(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lowercase:
                token = token.lower()
            split_tokens.extend(self._run_split_on_punc(token, self.never_split))
  
        output = whitespace_tokenize(''.join(split_tokens))
        
        return output
        
    
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

    def _is_punctuation(self, char):
        """Checks whether `char` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
    
    def _run_split_on_punc(self, text, never_split=None):
        if never_split is not None and text in never_split:
            return [text]
        
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append(char)
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                    start_new_word = False
                output[-1].append(char)
            i += 1
        
        return [''.join(x) for x in output]  # ['ni', ',' ,'hao']
            
    
    
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens





