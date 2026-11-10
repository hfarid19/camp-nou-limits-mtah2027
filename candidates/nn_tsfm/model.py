"""MOMENT-1 (AutonLab) adapted to per-step possession classification (torch; training only).

Architecture ("moment_tcn"):
  prepared input (T, Din) = nn_pff/train_tcn.prep (194 z-scored features + missing indicators)
    -> reduce:  Linear(Din -> Cm) per step, no bias            (Cm learned "channels", default 16)
    -> MOMENT:  RevIN per channel over the window, patches of 8 steps, value embedding + sinusoidal
                position embedding, flan-T5-small encoder (8 layers, d 512, 6 heads, channel-
                independent, weights from AutonLab/MOMENT-1-small)
    -> fuse:    Linear(512 -> dp) per channel token, concat over channels (Cm*dp), Linear -> Ch
    -> upsample x8 (repeat) to step rate, concat with skip = Linear(Din -> Cs) of the raw input
    -> head:    dilated residual TCN (Chh channels, dilations 1,2,4,8, k 3) -> Conv1d(-> 4)
  outputs (T, 4): H, A, D logits + hand-over logit (same contract as temporal_tcn / nn_pff_tcn).

The backbone is a plain transformers T5 encoder (T5EncoderModel.encoder); the MOMENT-specific parts
(RevIN, patching, embeddings) are re-implemented here from the momentfm source, because momentfm
does not install on this machine's Python 3.14. Only tensors the forward uses are loaded; T5's
token embedding (16.4 M) is dropped.

load_moment(path) -> (T5Config kwargs, state dict of the used tensors) from a local HF snapshot.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

PATCH = 8
REVIN_EPS = 1e-5
NORM_CLIP = 20.0


def load_moment(snapshot_dir):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(snapshot_dir, "config.json")))
    assert cfg["patch_len"] == PATCH and cfg["patch_stride_len"] == PATCH, cfg
    sd = load_file(os.path.join(snapshot_dir, "model.safetensors"))
    t5 = cfg["t5_config"]
    kw = dict(d_model=t5["d_model"], d_kv=t5["d_kv"], d_ff=t5["d_ff"], num_layers=t5["num_layers"],
              num_heads=t5["num_heads"], relative_attention_num_buckets=t5["relative_attention_num_buckets"],
              relative_attention_max_distance=t5["relative_attention_max_distance"],
              dropout_rate=t5["dropout_rate"], layer_norm_epsilon=t5["layer_norm_epsilon"],
              feed_forward_proj=t5["feed_forward_proj"])
    used = {k: v for k, v in sd.items() if k.startswith("encoder.") and "embed_tokens" not in k}
    used["value_embedding.weight"] = sd["patch_embedding.value_embedding.weight"]
    used["pe"] = sd["patch_embedding.position_embedding.pe"][0]          # (5000, d)
    return kw, used


def t5_encoder(kw, dropout=None):
    from transformers import T5Config, T5EncoderModel
    cfg = T5Config(vocab_size=8, is_encoder_decoder=False, use_cache=False, **kw)
    if dropout is not None:
        cfg.dropout_rate = dropout
    m = T5EncoderModel(cfg)
    return m.encoder          # T5Stack: block.*, final_layer_norm, embed_tokens (unused)


class Block(nn.Module):
    def __init__(self, C, d, k, drop):
        super().__init__()
        self.a = nn.Conv1d(C, C, k, dilation=d, padding=d * (k // 2))
        self.b = nn.Conv1d(C, C, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.b(self.drop(Fn.relu(self.a(Fn.relu(x)))))


class MomentTCN(nn.Module):
    def __init__(self, din, t5_kw, Cm=16, dp=32, Ch=96, Cs=64, Chh=96, dils=(1, 2, 4, 8), drop=0.1,
                 in_drop=0.1, t5_dropout=0.1, pe=None, max_patches=5000):
        super().__init__()
        d = t5_kw["d_model"]
        self.Cm, self.d = Cm, d
        self.in_drop = nn.Dropout(in_drop)
        self.reduce = nn.Linear(din, Cm, bias=False)
        self.value_embedding = nn.Linear(PATCH, d, bias=False)
        if pe is None:
            pe = torch.zeros(max_patches, d)
            pos = torch.arange(max_patches).float()[:, None]
            div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.clone())
        self.emb_drop = nn.Dropout(t5_dropout)
        self.encoder = t5_encoder(t5_kw, t5_dropout)
        self.proj = nn.Linear(d, dp)
        self.fuse = nn.Linear(Cm * dp, Ch)
        self.skip = nn.Conv1d(din, Cs, 1)
        self.inp = nn.Conv1d(Ch + Cs, Chh, 1)
        self.blocks = nn.ModuleList([Block(Chh, dd, 3, drop) for dd in dils])
        self.out = nn.Conv1d(Chh, 4, 1)
        self.dils = list(dils)

    def load_pretrained(self, used):
        enc = {k[len("encoder."):]: v for k, v in used.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        missing = [k for k in missing if "embed_tokens" not in k]
        assert not missing and not unexpected, (missing, unexpected)
        self.value_embedding.weight.data.copy_(used["value_embedding.weight"])
        self.pe.data.copy_(used["pe"][: self.pe.shape[0]])

    def forward(self, x, valid=None):
        """x (B, Din, T) prepared input, T a multiple of PATCH; valid (B, T) bool (False = padding,
        excluded from RevIN statistics and attention) -> (B, 4, T) logits."""
        B, Din, T = x.shape
        assert T % PATCH == 0, T
        n = T // PATCH
        xd = self.in_drop(x)
        s = self.reduce(xd.transpose(1, 2)).transpose(1, 2)                   # (B, Cm, T)
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        vm = valid[:, None, :].to(s.dtype)                                    # (B, 1, T)
        cnt = vm.sum(-1, keepdim=True).clamp(min=1.0)
        mean = (s * vm).sum(-1, keepdim=True) / cnt
        var = (((s - mean) * vm) ** 2).sum(-1, keepdim=True) / cnt
        z = ((s - mean) / torch.sqrt(var + REVIN_EPS)).clamp(-NORM_CLIP, NORM_CLIP) * vm
        p = z.reshape(B, self.Cm, n, PATCH)                                   # patches
        e = self.value_embedding(p) + self.pe[:n][None, None]                 # (B, Cm, n, d)
        e = self.emb_drop(e).reshape(B * self.Cm, n, self.d)
        pv = valid.reshape(B, n, PATCH).all(-1)                               # patch valid (B, n)
        am = pv.repeat_interleave(self.Cm, 0).to(e.dtype)                     # (B*Cm, n)
        h = self.encoder(inputs_embeds=e, attention_mask=am).last_hidden_state
        h = h.reshape(B, self.Cm, n, self.d).permute(0, 2, 1, 3)              # (B, n, Cm, d)
        f = self.fuse(self.proj(h).reshape(B, n, self.Cm * self.proj.out_features))  # (B, n, Ch)
        f = f.transpose(1, 2).repeat_interleave(PATCH, -1)                     # (B, Ch, T)
        y = self.inp(torch.cat([f, self.skip(xd)], 1))
        for b in self.blocks:
            y = b(y)
        return self.out(Fn.relu(y))

    def param_groups(self, lr_backbone, lr_head, wd):
        bb, hd = [], []
        for name, p in self.named_parameters():
            (bb if name.startswith(("encoder.", "value_embedding.")) else hd).append(p)
        return [dict(params=bb, lr=lr_backbone, weight_decay=wd), dict(params=hd, lr=lr_head, weight_decay=wd)]

    def export(self):
        """-> dict of float32 numpy arrays for moment_np.forward (only tensors the forward uses)."""
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.state_dict().items()
              if "embed_tokens" not in k}
        sd["pe"] = sd["pe"][: 1024]
        return sd
