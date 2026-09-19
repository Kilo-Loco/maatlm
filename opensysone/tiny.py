"""A tiny, randomly initialised backbone + tokenizer built offline.

Used by the unit tests and by the CPU smoke test so the repo can be exercised
end to end without downloading weights. Never use this for a real model —
`scripts/train.sh` starts from a pretrained checkpoint.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from .model import SystemOneModel

_CORPUS = [
    "### State\n### Question\n### Statement\nCandidate answer: Candidate level: Answer (yes/no): yes no",
    "Is this candidate the correct answer? Does the state match this level? Is this statement true of the state?",
    "The quick brown fox jumps over the lazy dog. My running shoes arrived in the wrong size, can I swap them?",
    "billing shipping returns refund exchange delayed damaged angry calm frustrated positive negative neutral",
    "0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ .,;:!?()[]{}\"'-_/\\@#$%^&*+=<>|~`\n",
]


def tiny_tokenizer(vocab_size: int = 512) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_CORPUS * 4, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )


def tiny_model(
    hidden: int = 64,
    layers: int = 2,
    heads: int = 4,
    seed: int = 0,
    attn_implementation: str = "sdpa",
    save_to: Optional[str] = None,
) -> SystemOneModel:
    torch.manual_seed(seed)
    tokenizer = tiny_tokenizer()
    cfg = AutoConfig.for_model(
        "qwen3",
        vocab_size=len(tokenizer),
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=hidden // heads,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    cfg._attn_implementation = attn_implementation
    backbone = AutoModelForCausalLM.from_config(cfg)
    m = SystemOneModel(backbone, tokenizer, max_state_tokens=1024, model_name="opensysone-tiny")
    if save_to:
        m.save(save_to)
    return m
