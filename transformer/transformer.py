
import sys, os
import argparse
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import json as js
import random
from collections.abc import Iterable

from dataset_P3n9W31 import TP3n9W31Data
from dataset_simple import SimpleData

from wingoal_utils.common import (
    set_log_file,
    log,
    logs
)

sys.path.append('..')
from utils import dl_utils

set_log_file(os.path.split(__file__)[-1], timestamp=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_TYPE = 'transformer'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() / d_model * (-math.log(10000.)))
        pe[:, 0::2] = torch.sin(position * div_term) #+ torch.sin(1 * div_term) + torch.sin(2 * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #+ torch.cos(1 * div_term) + torch.cos(2 * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, attn_mask):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        d_k = d_v = d_model // n_heads
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.sdp_attn = ScaledDotProductAttention(dropout)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context = self.sdp_attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=512, d_ff=512):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)


class EncodeLayer(nn.Module):
    def __init__(self, corpora, dropout=0.):
        super(EncodeLayer, self).__init__()
        d_model = corpora.hp.hidden_units
        n_heads = corpora.hp.num_heads
        d_ff = corpora.hp.feed_forward_units
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_output = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecodeLayer(nn.Module):
    def __init__(self, corpora, dropout=0.):
        super(DecodeLayer, self).__init__()
        d_model = corpora.hp.hidden_units
        n_heads = corpora.hp.num_heads
        d_ff = corpora.hp.feed_forward_units
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Encoder(nn.Module):
    def __init__(self, corpora, dropout=0.):
        super(Encoder, self).__init__()
        d_model = corpora.hp.hidden_units
        n_layers = corpora.hp.num_layers
        self.src_emb = nn.Embedding(corpora.src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncodeLayer(corpora, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_outputs = self.dropout(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self, corpora, dropout=0.):
        super(Decoder, self).__init__()
        d_model = corpora.hp.hidden_units
        n_layers = corpora.hp.num_layers
        self.tgt_emb = nn.Embedding(corpora.tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecodeLayer(corpora, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1).to(device)
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs,
                                dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, corpora, name='', dropout=0.):
        super(Transformer, self).__init__()
        self.corpora = corpora
        self.name = name
        self.encoder = Encoder(corpora, dropout).to(device)
        self.decoder = Decoder(corpora, dropout).to(device)
        self.projection = nn.Linear(corpora.hp.hidden_units, corpora.tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logit = self.projection(dec_outputs)
        return dec_logit.view(-1, dec_logit.size(-1))

    def translate(self, enc_inputs):
        start_symbol_idx = self.corpora.tgt_vocab[self.corpora.start_symbol]
        terminate_symbol_idx = self.corpora.tgt_vocab[self.corpora.terminate_symbol]
        batch_size = enc_inputs.size(0)
        x_ = torch.LongTensor(enc_inputs).to(device)
        terminate_tag = torch.zeros(batch_size, 1).fill_(terminate_symbol_idx).long().to(device)
        dec_inputs = torch.zeros(batch_size, 1).fill_(start_symbol_idx).long().to(device)
        for i in range(self.corpora.max_tgt_seq_len):
            _logits = self(x_, dec_inputs)
            _logits = _logits.view(enc_inputs.size(0), -1, self.corpora.tgt_vocab_size)
            _, _preds = torch.max(_logits, dim=-1)
            if _preds[:, -1:].equal(terminate_tag):
                break
            dec_inputs = torch.cat([dec_inputs, _preds[:, -1:]], dim=-1)
        return dec_inputs[:, 1:]


def create_model(corpora, name='', dropout=0.):
    return Transformer(corpora, name, dropout)


# 从训练样本中抽 num_validate 个来验证翻译结果
def validate(model, num_validate=10, num_totals=0):
    corpora = model.corpora
    if num_totals <= 0:
        num_totals = len(corpora.enc_inputs)
    all_indices = list(range(num_totals))
    random.shuffle(all_indices)
    sample_indices = all_indices[:num_validate]
    src_sentences = [corpora.sources[i] for i in sample_indices]
    tgt_sentences = [corpora.targets[i] for i in sample_indices]
    enc_inputs = corpora.preprocess(src_sentences)
    translated = model.translate(enc_inputs)

    success = True
    passed, failed = 0, 0
    detail = []
    for src, tgt, pred in zip(src_sentences, tgt_sentences, translated):
        translated = corpora.to_tgt_sentence(pred.squeeze(0), first=True)
        tgt_std = ' '.join(tgt.split())
        if tgt_std == translated:
            passed += 1
        else:
            failed += 1
            detail.append({
                'source': src,
                'target': tgt,
                'translated': translated
            })
            success = False
    validate_result = {'success': success, 'num_validate': num_validate, 'passed': passed, 'failed': failed, 'detail': detail}
    return validate_result


# checkpoint的两种配置方式：
# 1. checkpoint_interval: 保存训练模型参数的轮数间隔，0或小于0，则不会保存模型结果到文件中
# 2. checkpoints: 指定保存哪些训练轮次的训练结果。提供此参数是基于解决模型存储空间的考虑
def train(model, loader, num_epochs, force_retrain=False, checkpoint_interval=5, checkpoints=None):
    if isinstance(checkpoints, list):
        checkpoints = iter(checkpoints)
    checkpoint = next(checkpoints) if isinstance(checkpoints, Iterable) else -1

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    last_epoch = 0
    model_name = model.name if model.name else 'default'
    # 如果不强制重训练，则尝试继承之前训练结果
    if not force_retrain:
        last_states = dl_utils.get_last_state(model_name, MODEL_TYPE, max_epoch=num_epochs)
        if not last_states:
            last_states = dl_utils.get_model_state(model_name, MODEL_TYPE)
        if last_states:
            model.load_state_dict(torch.load(last_states['file_name'], map_location=device))
            last_epoch = last_states['last_epoch']
    train_loss = []
    file_prefix = '-'.join([p for p in [MODEL_TYPE, model_name if model_name else 'default'] if p])
    for epoch in range(last_epoch, num_epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            log('epoch: {:4d}, loss = {:.6f}'.format(epoch + 1, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        if (checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0) \
                or epoch + 1 == checkpoint:
            # 保存 checkpoint
            file_name = 'state_dict\\%s-%s-model.pkl' % (file_prefix, epoch + 1)
            torch.save(model.state_dict(), file_name)
            loss_file = 'state_dict\\%s-%s-loss.json' % (file_prefix, epoch + 1)
            f = open(loss_file, 'w')
            f.write(js.dumps({'train_loss': train_loss[-100:]}, indent=2))
            f.close()
            if checkpoint > 0:
                try:
                    checkpoint = next(checkpoints)
                except StopIteration:
                    checkpoint = -1

    return model


def greedy_decoder(model, enc_input, start_symbol):
    corpora = model.corpora
    enc_outputs = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
                              -1)
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测时会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == corpora.terminate_symbol_idx:
            terminal = True
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


def predict_greedy(model, enc_inputs):
    corpora = model.corpora
    dec_outputs = []
    for enc_input in enc_inputs:
        dec_output = greedy_decoder(model, enc_input.view(
            1, -1).to(device), start_symbol=corpora.start_symbol_idx)
        dec_outputs.append(dec_output.squeeze(0))
    return dec_outputs


def predict_batch(model, enc_inputs):
    corpora = model.corpora
    x_ = torch.LongTensor(enc_inputs).to(device)
    batch_size = enc_inputs.size(0)
    terminate_tag = torch.ones(batch_size, 1) * corpora.terminate_symbol_idx
    terminate_tag = terminate_tag.to(device)
    start_inputs = torch.zeros((enc_inputs.size(0), 1)).fill_(corpora.start_symbol_idx).long().to(device)
    dec_inputs = start_inputs
    for i in range(corpora.max_tgt_seq_len):
        _logits = model(x_, dec_inputs)
        _logits = _logits.view(enc_inputs.size(0), -1, corpora.tgt_vocab_size)
        _, _preds = torch.max(_logits, dim=-1)
        if _preds[:, -1:].equal(terminate_tag):
            break
        dec_inputs = torch.cat([start_inputs, _preds], dim=-1)
    return dec_inputs[:, 1:]


def test_1(model):
    corpora = model.corpora
    enc_inputs = corpora.preprocess(corpora.demo_sentences)
    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子 翻译成英文句子: ")
    dec_outputs = predict_greedy(model, enc_inputs)
    i = 0
    for enc_input, greedy_dec_predict in zip(enc_inputs, dec_outputs):
        print('%s.' % (i + 1), corpora.to_src_sentence(enc_input, ''))
        print('->', corpora.to_tgt_sentence(greedy_dec_predict), '\n')
        i += 1


def test_2(model):
    corpora = model.corpora
    enc_inputs = corpora.preprocess(corpora.demo_sentences)
    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子 翻译成英文句子: ")
    dec_outputs = predict_batch(model, enc_inputs)
    i = 0
    for enc_input, pred in zip(enc_inputs, dec_outputs):
        print('%s.' % (i + 1), corpora.to_src_sentence(enc_input, ''))
        print('->', corpora.to_tgt_sentence(pred.squeeze()), '\n')
        i += 1


def test(model):
    # 1. test with greedy_decoder
    # test_1(model)
    # 2. test with normal method
    test_2(model)


def main():
    # corpora, name, batch_size, num_epochs = TP3n9W31Data(), 'classical', 48, 800
    corpora, name, batch_size, num_epochs = SimpleData(), 'simple', 4, 200
    _dropout = 0.1
    loader = Data.DataLoader(corpora, batch_size, True)
    model = Transformer(corpora, name, dropout=_dropout).to(device)
    train(model, loader, num_epochs, force_retrain=False)
    model.eval()
    # score_list = evaluate(model)
    # score_table = AsciiTable(score_list)
    # log("\n" + score_table.table)

    log('validating model ...')
    result = validate(model, 100)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))

    log('test demo sentences ...')
    test(model)


def show_net():
    corpora = SimpleData()
    _dropout = 0.1
    net = Transformer(corpora, 'simple_show_net', dropout=_dropout).to(device)
    param_count, net_info = dl_utils.get_net_detail(net, show_param_shape=True)
    logs('model detail:', net_info)
    log('total parameters: ', param_count)


def state_dict():
    corpora = SimpleData()
    _dropout = 0.1
    net = Transformer(corpora, 'simple_validate', dropout=_dropout).to(device)
    param_count = 0
    for k, v in net.state_dict().items():
        num_el = v.numel()
        param_count += num_el
        print(k, '[%s]' % num_el)
    log('total parameters:', param_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params", nargs="*")
    args = parser.parse_args()
    if len(args.params) == 0:
        log('executing function [main] ...')
        main()
    else:
        func = args.params[0]
        if func != 'main':
            set_log_file(os.path.split(__file__)[-1], suffix=func, timestamp=True)
        param_list = args.params[1:]
        log('executing function [%s] ...' % func)
        eval(func)(*param_list)
    log('finish executing function!')

