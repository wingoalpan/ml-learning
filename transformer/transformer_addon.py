import os, sys
import torch
import torch.nn as nn
from transformer import Transformer

sys.path.append('..\\utils')
import dl_utils
sys.path.append('..\\..\\wingoal_utils')
from common import log


class AdaptiveLinear(nn.Module):
    use_adapter = True

    def __init__(self, base_linear: nn.Module, rank=4):
        super(AdaptiveLinear, self).__init__()
        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        has_bias = base_linear.bias is not None
        device = base_linear.weight.device

        self.lora_a = nn.Linear(in_features, rank, bias=False).to(device)
        self.lora_b = nn.Linear(rank, out_features, bias=has_bias).to(device)
        # nn.init.constant_(self.lora_a.weight, 0.0)
        # nn.init.constant_(self.lora_b.weight, 0.0)

        for p in self.base_linear.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if AdaptiveLinear.use_adapter:
            return self.base_linear(x) + self.lora_b(self.lora_a(x))
        else:
            return self.base_linear(x)


def merge_vocab(model, corpora):
    base_corpora = model.corpora
    pre_src_vocab_size = len(base_corpora.src_vocab)
    pre_tgt_vocab_size = len(base_corpora.tgt_vocab)

    last_idx = len(base_corpora.src_vocab)
    for word in corpora.src_vocab.keys():
        if word not in base_corpora.src_vocab.keys():
            base_corpora.src_vocab[word] = last_idx
            last_idx += 1

    last_idx = len(base_corpora.tgt_vocab)
    for word in corpora.tgt_vocab.keys():
        if word not in base_corpora.tgt_vocab.keys():
            base_corpora.tgt_vocab[word] = last_idx
            last_idx += 1

    base_corpora.src_vocab_size = len(base_corpora.src_vocab)
    base_corpora.tgt_vocab_size = len(base_corpora.tgt_vocab)
    base_corpora.src_idx2w = {v: k for k, v in base_corpora.src_vocab.items()}
    base_corpora.tgt_idx2w = {v: k for k, v in base_corpora.tgt_vocab.items()}

    inc_src_vocab_size = base_corpora.src_vocab_size - pre_src_vocab_size
    inc_tgt_vocab_size = base_corpora.tgt_vocab_size - pre_tgt_vocab_size

    if inc_src_vocab_size > 0:
        device = model.encoder.src_emb.weight.device
        embedding_dim = model.encoder.src_emb.embedding_dim
        inc_emb = nn.Embedding(inc_src_vocab_size, embedding_dim).to(device)
        model.encoder.src_emb.num_embeddings = base_corpora.src_vocab_size
        model.encoder.src_emb.weight.data = torch.cat((model.encoder.src_emb.weight.data, inc_emb.weight.data), dim=0)

    if inc_tgt_vocab_size > 0:
        device = model.decoder.tgt_emb.weight.device
        embedding_dim = model.decoder.tgt_emb.embedding_dim
        inc_emb = nn.Embedding(inc_tgt_vocab_size, embedding_dim).to(device)
        model.decoder.tgt_emb.num_embeddings = base_corpora.tgt_vocab_size
        model.decoder.tgt_emb.weight.data = torch.cat((model.decoder.tgt_emb.weight.data, inc_emb.weight.data), dim=0)

        in_features = model.projection.in_features
        inc_projection = nn.Linear(in_features, inc_tgt_vocab_size).to(device)
        model.projection.out_features = base_corpora.tgt_vocab_size
        model.projection.weight.data = torch.cat((model.projection.weight.data, inc_projection.weight.data), dim=0)

    return inc_src_vocab_size, inc_tgt_vocab_size


def add_adapter(model, rank):
    adapters = {}
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            adapters[name] = AdaptiveLinear(module)
    for name, module in adapters.items():
        model._modules[name] = module

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, AdaptiveLinear):
            continue
        if dl_utils._has_children(module):
            add_adapter(module, rank)
        elif not isinstance(module, nn.Embedding):
            for p in module.parameters():
                p.requires_grad_(False)

    return model
