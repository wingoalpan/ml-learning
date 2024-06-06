
import sys, os
import argparse
import torch
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
sys.path.append('..\\utils')
import dl_utils

set_log_file(os.path.split(__file__)[-1], timestamp=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_TYPE = 'lstm'


class Encoder(nn.Module):
    def __init__(self, corpora, dropout=0.):
        super(Encoder, self).__init__()
        d_model = corpora.hp.hidden_units
        n_layers = corpora.hp.num_lstm_layers
        d_ff = corpora.hp.feed_forward_units
        self.src_emb = nn.Embedding(corpora.src_vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_ff, num_layers=n_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, enc_inputs):
        enc_inputs = self.src_emb(enc_inputs)
        enc_outputs, (h, c) = self.lstm(enc_inputs)
        return enc_outputs, (h ,c)


class Decoder(nn.Module):
    def __init__(self, corpora, dropout=0.):
        super(Decoder, self).__init__()
        d_model = corpora.hp.hidden_units
        n_layers = corpora.hp.num_lstm_layers
        d_ff = corpora.hp.feed_forward_units
        self.tgt_emb = nn.Embedding(corpora.tgt_vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_ff, num_layers=n_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, dec_inputs, h, c):
        dec_inputs = self.tgt_emb(dec_inputs)
        dec_outputs, (h, c) = self.lstm(dec_inputs, (h, c))
        return dec_outputs, (h ,c)


class Seq2Seq(nn.Module):
    def __init__(self, corpora, name='', dropout_enc=0., dropout_dec=0.):
        super(Seq2Seq, self).__init__()
        self.corpora = corpora
        self.name = name
        self.encoder = Encoder(corpora, dropout_enc).to(device)
        self.decoder = Decoder(corpora, dropout_dec).to(device)
        self.projection = nn.Linear(corpora.hp.hidden_units, corpora.tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs, teach_rate=0.5):
        batch_size = enc_inputs.size(0)
        dec_seq_len = dec_inputs.size(1)
        outputs = torch.zeros(batch_size, dec_seq_len, self.corpora.tgt_vocab_size)

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
        enc_inputs = enc_inputs.to(device)
        start_symbol_idx = self.corpora.tgt_vocab[self.corpora.start_symbol]
        batch_size = enc_inputs.size(0)
        outputs = torch.zeros(batch_size, self.corpora.max_tgt_seq_len).to(device)

        _, (h, c) = self.encoder(enc_inputs)
        dec_input_slice = torch.zeros(batch_size, 1).fill_(start_symbol_idx).long().to(device)
        for i in range(0, self.corpora.max_tgt_seq_len):
            dec_output, (h, c) = self.decoder(dec_input_slice, h, c)
            dec_output = self.projection(dec_output)
            top = dec_output.argmax(-1)
            outputs[:, i] = top[:, 0]
            dec_input_slice = top
        return outputs


def create_model(corpora, name='', dropout=0.):
    return Seq2Seq(corpora, name, dropout, dropout)


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
        if tgt == translated:
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

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
            outputs = outputs.reshape(-1, model.corpora.tgt_vocab_size).to(device)
            loss = criterion(outputs, dec_outputs.view(-1))
            log('epoch: {:4d}, loss = {:.6f}'.format(epoch + 1, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        if (checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0)\
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


def test(model):
    corpora = model.corpora
    enc_inputs = corpora.preprocess(corpora.demo_sentences).to(device)
    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子 翻译成英文句子: ")
    dec_outputs = model.translate(enc_inputs)
    i = 0
    for enc_input, predict in zip(enc_inputs, dec_outputs):
        print('%s.' % (i + 1), corpora.to_src_sentence(enc_input, ''))
        print('->', corpora.to_tgt_sentence(predict), '\n')
        i += 1


def main():
    # corpora = TP3n9W31Data()
    corpora = SimpleData()
    batch_size = 4
    _dropout = 0.
    loader = Data.DataLoader(corpora, batch_size, True)
    model = Seq2Seq(corpora, 'simple_dropout0.3', _dropout, _dropout).to(device)
    train(model, loader, 50, force_retrain=False)
    log('validating ...')
    result = validate(model)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))

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
            set_log_file(os.path.split(__file__)[-1], suffix=func, timestamp=True)
        param_list = args.params[1:]
        log('executing function [%s] ...' % func)
        eval(func)(*param_list)
    log('finish executing function!')

