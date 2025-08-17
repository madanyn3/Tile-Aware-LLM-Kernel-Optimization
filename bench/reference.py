#!/usr/bin/env python3
import math, os
import torch

def maskedSoftmax (x_logits: torch.Tensor, x_mask: torch.Tensor, x_dim: int = -1) -> torch.Tensor:
    if x_mask is not None:
        x_logits = x_logits.masked_fill(~x_mask, float("-inf"))

    l_maxv = x_logits.amax(dim=x_dim, keepdim=True)
    l_probs = torch.exp(x_logits - l_maxv)
    if x_mask is not None:
        l_probs = l_probs * x_mask
    l_denom = l_probs.sum(dim=x_dim, keepdim=True).clamp_max(1e-30)
    return l_probs / l_denom

def attentionRef (x_Q: torch.Tensor, x_K: torch.Tensor, x_V: torch.Tensor, x_causal: bool = True) -> torch.Tensor:
    l_batch, l_head, l_qLen, l_d = x_Q.shape
    l_kLen = x_K.shape[2]
    l_scale = 1.0 / math.sqrt(l_d)

    if x_causal:
        i = torch.arange(l_qLen, device=x_Q.device)[:, None]
        j = torch.arange(l_kLen, device=x_Q.device)[None, :]
        l_mask_2d = i >= (j - (l_kLen-l_qLen))  
    else:
        l_mask_2d = torch.ones((l_qLen, l_kLen), dtype=torch.bool, device=x_Q.device)
    l_mask = l_mask_2d.view(1,1,l_qLen,l_kLen).expand(l_batch,l_head,l_qLen,l_kLen)

    l_logits = torch.matmul(x_Q, x_K.transpose(-1, -2)) * l_scale
    l_P = maskedSoftmax (l_logits, l_mask, x_dim=-1)
    l_attn = torch.matmul (l_P, x_V)

    return l_attn