
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

from dataset_P3n9W31 import TP3n9W31Data
from dataset_simple import SimpleData

sys.path.append('..\\utils')
sys.path.append('..\\..\\wingoal_utils')
from common import (
    set_log_file,
    log,
    logs
)
import dl_utils

set_log_file(os.path.split(__file__)[-1], timestamp=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_TYPE = 'lstm'

ds = TP3n9W31Data()
# ds = SimpleData()

n_layers = ds.hyperparams.get('num_lstm_layers', 4)
d_model = ds.hyperparams.get('hidden_units', 512)
d_ff = ds.hyperparams.get('feed_forward_units', 512)
num_epochs = ds.hyperparams.get('num_epochs', 400)
batch_size = ds.hyperparams.get('batch_size', 32)

d_k = d_v = d_model // n_heads

loader = Data.DataLoader(ds, batch_size, True)


class Encoder(nn.Module):
    def __init__(self, dropout=0.):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(ds.src_vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_ff, num_layers=n_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, enc_inputs):
        enc_inputs = self.src_emb(enc_inputs)
        enc_outputs, (h, c) = self.lstm(enc_inputs)
        return enc_outputs, (h ,c)


class Decoder(nn.Module):
    def __init__(self, dropout=0.):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(ds.tgt_vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_ff, num_layers=n_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, dec_inputs, h, c):
        dec_inputs = self.tgt_emb(dec_inputs)
        dec_outputs, (h, c) = self.lstm(dec_inputs, (h, c))
        return dec_outputs, (h ,c)


class Seq2Seq(nn.Module):
    def __init__(self, dropout_enc=0., dropout_dec=0.):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(dropout_enc).to(device)
        self.decoder = Decoder(dropout_dec).to(device)
        self.projection = nn.Linear(d_model, ds.tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs, teach_rate=0.5):
        _batch_size = enc_inputs.size(0)
        dec_seq_len = dec_inputs.size(1)
        outputs = torch.zeros(_batch_size, dec_seq_len, ds.tgt_vocab_size)

        _, (h, c) = self.encoder(enc_inputs)
        dec_input_slice = dec_inputs[:, 0].view(-1, 1)
        for i in range(0, dec_seq_len):
            dec_output, (h, c) = self.decoder(dec_input_slice, h, c)
            dec_output = self.projection(dec_output)
            outputs[:, i, :] = dec_output[:, 0, :]
            teach_prob =  random.random()
            top = dec_output.argmax(-1)
            if i < dec_seq_len - 1:   # 准备下一个时间序的 dec input
                dec_input_slice = dec_inputs[:, i + 1].view(-1, 1) if teach_prob < teach_rate else top
        return outputs

    def translate(self, enc_inputs):
        _batch_size = enc_inputs.size(0)
        outputs = torch.zeros(_batch_size, ds.max_tgt_seq_len).to(device)

        _, (h, c) = self.encoder(enc_inputs)
        dec_input_slice = torch.zeros(_batch_size, 1).fill_(ds.tgt_vocab[ds.start_symbol]).long().to(device)
        for i in range(0, ds.max_tgt_seq_len):
            dec_output, (h, c) = self.decoder(dec_input_slice, h, c)
            dec_output = self.projection(dec_output)
            top = dec_output.argmax(-1)
            outputs[:, i] = top[:, 0]
            dec_input_slice = top
        return outputs


# 从训练样本中抽 num_validate 个来验证翻译结果
def validate(model, num_totals=0):
    if num_totals <= 0:
        num_totals = len(ds.enc_inputs)
    num_validate = ds.hyperparams.get('num_validate', 10)
    all_indices = list(range(num_totals))
    random.shuffle(all_indices)
    sample_indices = all_indices[:num_validate]
    src_sentences = [ds.sources[i] for i in sample_indices]
    tgt_sentences = [ds.targets[i] for i in sample_indices]
    enc_inputs = ds.preprocess(src_sentences).to(device)
    log('translating ...')
    translated = model.translate(enc_inputs)
    i = 0
    for src, tgt, pred in zip(src_sentences, tgt_sentences, translated):
        print('%s.' % (i + 1), src)
        print('==', tgt)
        print('->', ' '.join([ds.tgt_idx2w[n.item()] for n in pred.squeeze()]).split(ds.terminate_symbol)[0], '\n')
        i += 1


def train(model):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    last_epoch = 0
    model_name = ds.hyperparams.get('model_name', 'default')
    last_states = dl_utils.get_last_state(model_name, MODEL_TYPE, max_epoch=num_epochs)
    if not last_states:
        last_states = dl_utils.get_model_state(model_name, MODEL_TYPE)
    if last_states:
        model.load_state_dict(torch.load(last_states['file_name'], map_location=torch.device('cpu')))
        last_epoch = last_states['last_epoch']
    train_loss = []
    file_prefix = '-'.join([p for p in [MODEL_TYPE, model_name if model_name else 'default'] if p])
    for epoch in range(last_epoch, num_epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)
            outputs = model(enc_inputs, dec_inputs)
            outputs = outputs.reshape(-1, ds.tgt_vocab_size).to(device)
            loss = criterion(outputs, dec_outputs.view(-1))
            log('epoch: {:4d}, loss = {:.6f}'.format(epoch + 1, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        if (epoch + 1) % 5 == 0:
            # 保存 checkpoint
            file_name = 'state_dict\\%s-%s-model.pkl' % (file_prefix, epoch + 1)
            torch.save(model.state_dict(), file_name)
            loss_file = 'state_dict\\%s-%s-loss.json' % (file_prefix, epoch + 1)
            f = open(loss_file, 'w')
            f.write(js.dumps({'train_loss': train_loss[-100:]}, indent=2))
            f.close()

    return model


def test(model):
    sentences = ds.demo_sentences
    enc_inputs = ds.preprocess(sentences).to(device)
    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子 翻译成英文句子: ")
    dec_outputs = model.translate(enc_inputs)
    i = 0
    for enc_input, greedy_dec_predict in zip(enc_inputs, dec_outputs):
        print('%s.' % (i + 1), ''.join([ds.src_idx2w[t.item()] for t in enc_input if t > 0]))
        print('->', ' '.join([ds.tgt_idx2w[n.item()] for n in greedy_dec_predict]).split(ds.terminate_symbol)[0], '\n')
        i += 1


def main():
    _dropout = ds.hyperparams.get('dropout', 0.4)
    model = Seq2Seq(_dropout, _dropout).to(device)
    train(model)
    validate(model)
    test(model)


def show_net():
    net = Seq2Seq()
    param_count, net_info = dl_utils.get_net_detail(net, show_param_shape=True)
    logs('model detail:', net_info)
    log('total parameters: ', param_count)


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
            CM.set_log_file(os.path.split(__file__)[-1], suffix=func, timestamp=True)
        param_list = args.params[1:]
        log('executing function [%s] ...' % func)
        eval(func)(*param_list)
    log('finish executing function!')

