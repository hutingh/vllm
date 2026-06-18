# coding=utf-8
# Copyright (c) The Pangu team and The HuggingFace Inc. team.
#
# This code keeps SophonTokenizerFast loadable with both Transformers 4.x and
# Transformers 5.x. Transformers 5.x uses TokenizersBackend, while 4.x still
# uses PreTrainedTokenizerFast.

"""Tokenization Fast class for Sophon."""

from typing import Dict, List, Optional

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from transformers.utils import logging

try:
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
except ImportError:
    TokenizersBackend = None
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

PRETOKENIZE_REGEX = r"'(?i:[sdmt]|ll|ve|re)| (?=\p{Han}|[＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。])|[＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。]+[\r\n]*|[^\r\n\p{L}\p{N}]?+[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"


if TokenizersBackend is not None:

    class SophonTokenizerFast(TokenizersBackend):
        vocab_files_names = VOCAB_FILES_NAMES
        padding_side = "left"
        model_input_names = ["input_ids", "attention_mask"]
        model = BPE

        def __init__(
            self,
            vocab: Optional[Dict[str, int]] = None,
            merges: Optional[List[str]] = None,
            tokenizer_file: Optional[str] = None,
            bos_token: str = "<|pangu_text_start|>",
            eos_token: str = "<|pangu_text_end|>",
            unk_token: str = None,
            add_bos_token: bool = True,
            add_eos_token: bool = False,
            add_prefix_space: bool = False,
            **kwargs,
        ):
            self._vocab = vocab or {}
            self._merges = merges or []
            self._add_bos_token = add_bos_token
            self._add_eos_token = add_eos_token

            if tokenizer_file is not None and vocab is None and merges is None:
                self._tokenizer = Tokenizer.from_file(tokenizer_file)
            else:
                self._tokenizer = Tokenizer(
                    BPE(
                        vocab=self._vocab,
                        merges=self._merges,
                        dropout=None,
                        unk_token=unk_token,
                        byte_fallback=True,
                    )
                )
                self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                    pre_tokenizers.Split(
                        Regex(PRETOKENIZE_REGEX),
                        behavior="isolated",
                    ),
                    pre_tokenizers.ByteLevel(
                        add_prefix_space=add_prefix_space,
                        use_regex=False,
                    ),
                ])
                self._tokenizer.decoder = decoders.ByteLevel()

            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
                **kwargs,
            )

            self.update_post_processor()

        def update_post_processor(self):
            bos = self.bos_token
            bos_id = self.bos_token_id
            if bos is None and self.add_bos_token:
                raise ValueError("add_bos_token = True but bos_token = None")

            eos = self.eos_token
            eos_id = self.eos_token_id
            if eos is None and self.add_eos_token:
                raise ValueError("add_eos_token = True but eos_token = None")

            single = f"{bos}:0 $A:0" if self._add_bos_token else "$A:0"
            if self._add_eos_token:
                single += f" {eos}:0"

            special_tokens = []
            if self._add_bos_token:
                special_tokens.append((bos, bos_id))
            if self._add_eos_token:
                special_tokens.append((eos, eos_id))

            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=single,
                pair=f"{single} {single.replace('$A', '$B')}",
                special_tokens=special_tokens,
            )

        @property
        def add_bos_token(self):
            return self._add_bos_token

        @add_bos_token.setter
        def add_bos_token(self, value):
            self._add_bos_token = value
            self.update_post_processor()

        @property
        def add_eos_token(self):
            return self._add_eos_token

        @add_eos_token.setter
        def add_eos_token(self, value):
            self._add_eos_token = value
            self.update_post_processor()

        @property
        def vocab_size(self):
            return self._tokenizer.get_vocab_size(with_added_tokens=True)

else:

    class SophonTokenizerFast(PreTrainedTokenizerFast):
        vocab_files_names = VOCAB_FILES_NAMES
        padding_side = "left"
        model_input_names = ["input_ids", "attention_mask"]
        _auto_class = "AutoTokenizer"

        def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            bos_token="<|pangu_text_start|>",
            eos_token="<|pangu_text_end|>",
            add_bos_token=True,
            add_eos_token=False,
            decode_with_prefix_space=False,
            clean_up_tokenization_spaces=False,
            **kwargs,
        ):
            super().__init__(
                vocab_file=vocab_file,
                tokenizer_file=tokenizer_file,
                bos_token=bos_token,
                eos_token=eos_token,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
                decode_with_prefix_space=decode_with_prefix_space,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            self._add_bos_token = add_bos_token
            self._add_eos_token = add_eos_token
            self.tokenizer_file = tokenizer_file
            self.update_post_processor()

        def update_post_processor(self):
            bos = self.bos_token
            bos_token_id = self.bos_token_id
            if bos is None and self.add_bos_token:
                raise ValueError("add_bos_token = True but bos_token = None")

            eos = self.eos_token
            eos_token_id = self.eos_token_id
            if eos is None and self.add_eos_token:
                raise ValueError("add_eos_token = True but eos_token = None")

            single = f"{(bos + ':0 ') if self.add_bos_token else ''}$A:0{(' ' + eos + ':0') if self.add_eos_token else ''}"
            pair = f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} $B:1{(' ' + eos + ':1') if self.add_eos_token else ''}"

            special_tokens = []
            if self.add_bos_token:
                special_tokens.append((bos, bos_token_id))
            if self.add_eos_token:
                special_tokens.append((eos, eos_token_id))
            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=single, pair=pair, special_tokens=special_tokens
            )

        @property
        def add_eos_token(self):
            return self._add_eos_token

        @property
        def add_bos_token(self):
            return self._add_bos_token

        @add_eos_token.setter
        def add_eos_token(self, value):
            self._add_eos_token = value
            self.update_post_processor()

        @add_bos_token.setter
        def add_bos_token(self, value):
            self._add_bos_token = value
            self.update_post_processor()

        @property
        def vocab_size(self):
            return self.backend_tokenizer.get_vocab_size()

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            if self._add_bos_token:
                bos_token_ids = [self.bos_token_id]
            else:
                bos_token_ids = []

            output = bos_token_ids + token_ids_0

            if token_ids_1 is not None:
                output = output + token_ids_1

            if self.add_eos_token:
                output = output + [self.eos_token_id]

            return output


__all__ = ["SophonTokenizerFast"]
