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
import logging

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union


logger = logging.getLogger(__name__)

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
                 do_lowercase=False,
                 do_basic_tokenize=True,
                 tokenize_chinese=True,
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
        self.never_split = never_split if never_split is not None else []
        self.do_basic_tokenize = do_basic_tokenize
        
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        
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
               add_special_tokens: bool = False,
               max_length: Optional[int] = None,
               padding='max_length'
              # padding_strategy='max_length',
               #truncation_strategy='only_first',
               ):
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self._tokenize(text)
            return self.convert_tokens2ids(tokens)
        
        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        
        return self.prepare_for_model(
            ids=first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding
        )
    
    def convert_tokens2ids(self, tokens):
        if tokens is None:
            return None
        if isinstance(tokens, str):
            if tokens in self.tokens2ids:
                return self.tokens2ids[tokens]
    
        ids = []
        for token in tokens:
            ids.append(self._convert_token2id(token))
        return ids
    
    def _convert_token2id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokens2ids.get(token, self.tokens2ids.get(self.unk_token))

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))
    
    def prepare_for_model(self,
                          ids: List[int],
                          pair_ids: Optional[List[int]] = None,
                          add_special_tokens: bool = False,
                          max_length: Optional[int] =None,
                          padding: Union[bool, str] = True,
                          return_token_type_ids: bool = True,
                          return_attention_mask: bool = True
                          ):
        
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        
        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        
        if max_length and total_len > max_length:
            ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy='ONLY_FIRST',  # first是针对ids，second是针对pair_ids
                direction='RIGHT'
            )

        # Add special tokens
        if add_special_tokens:
            print('ids:{}'.format(ids))
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        print('encoded_inputs:{}'.format(encoded_inputs))
        
        # Padding
        if padding or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy='DO_NOT_PAD',
                return_attention_mask=return_attention_mask
            )
        
        return encoded_inputs
    
    def pad(self,
            encoded_inputs,
            max_length: Optional[int] = None,
            padding_strategy='MAX_LENGTH',
            return_attention_mask: bool = True
            ):
        
        if encoded_inputs["input_ids"] and not isinstance(encoded_inputs["input_ids"][0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask,
            )
            return encoded_inputs
        
        batch_size = len(encoded_inputs["input_ids"])
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."
        
        if padding_strategy == 'LONGEST':
            max_length = max(len(inputs) for inputs in encoded_inputs["input_ids"])
            padding_strategy = 'MAX_LENGTH'

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask,
            )
    
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
                
        return batch_outputs
    
    def truncate_sequences(self,
                           ids: List[int],
                           pair_ids: Optional[List[int]] = None,
                           num_tokens_to_remove: int = 0,
                           truncation_strategy: str = 'ONLY_FIRST',
                           direction: str ='RIGHT',
                           stride: int = 0
                           ):
        
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        overflowing_tokens = []
        if truncation_strategy == 'ONLY_FIRST':
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if direction == 'RIGHT':
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                elif direction == 'LEFT':
                    overflowing_tokens = ids[:window_len]
                    ids = ids[window_len:]
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input"
                    f"but the first sequence has a length {len(ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    f"for instance 'longest_first' or 'only_second'."
                )
        else:
            logger.error(f"Please select correct truncation strategy, for instance 'ONLY_FIRST'")
        
        return ids, overflowing_tokens

    def _pad(self, encoded_inputs, max_length=None, padding_strategy=None,
             return_attention_mask: Optional[bool] = None):
        print(encoded_inputs)
        needs_to_be_padded = padding_strategy != 'DO_NOT_PAD' and len(encoded_inputs["input_ids"]) != max_length
        if needs_to_be_padded:
            if padding_strategy == "MAX_LENGTH":
                difference = max_length - len(encoded_inputs["input_ids"])
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                    encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            else:
                raise ValueError("Invalid padding strategy")
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
    
        return encoded_inputs

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    @property
    def sep_token_id(self) -> Optional[int]:
       
        if self.sep_token is None:
            return None
        return self.convert_tokens2ids(self.sep_token)

    @property
    def cls_token_id(self) -> Optional[int]:

        if self.cls_token is None:
            return None
        return self.convert_tokens2ids(self.cls_token)

    @property
    def pad_token_id(self) -> Optional[int]:
    
        if self.pad_token is None:
            return None
        return self.convert_tokens2ids(self.pad_token)

class BasicTokenizer():  # 没必要继承object类，因为python3默认继承object类，里面有很多高级特性
    
    def __init__(self, do_lowercase=True, tokenize_chinese=False, never_split=None):
        self.do_lowercase = do_lowercase
        self.tokenize_chinese = tokenize_chinese
        self.never_split = never_split
        
    def tokenize(self, text):
        text = self._clean_text(text)
        
        if self.tokenize_chinese:
            text = self._tokenize_chinese(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lowercase:
                token = token.lower()
            split_tokens.extend(self._run_split_on_punc(token, self.never_split))
  
        output = whitespace_tokenize(' '.join(split_tokens))
        
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





