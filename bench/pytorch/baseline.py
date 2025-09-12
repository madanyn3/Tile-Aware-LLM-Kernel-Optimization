#!/usr/bin/env python3

import math, os
import torch

def attentionSDPA (x_Q: torch.Tensor, x_K: torch.Tensor, x_V: torch.Tensor, x_causal: bool = True) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        x_Q, x_K, x_V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=x_causal
    )