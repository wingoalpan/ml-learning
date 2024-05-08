import torch
import torch.utils.data as Data
import json as js


# 定义语言翻译数据集的通用接口
class TransDataset(Data.Dataset):
    def __init__(self, hyper_params=None):
        self.sources = []
        self.targets = []
        self.src_vocab = dict()  # source word's idx
        self.tgt_vocab = dict()  # target word's idx
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        self.src_idx2w = dict()  # idx to source word
        self.tgt_idx2w = dict()  # idx to target word
        self.pad_symbol = '<PAD>'
        self.unknown_symbol = '<UNK>'
        self.terminate_symbol = '</S>'
        self.start_symbol = '<S>'
        self.terminate_symbol_idx = 3
        self.start_symbol_idx = 2
        self.terminate_symbol = '</S>'
        self.max_src_seq_len = 50
        self.max_tgt_seq_len = 50
        self.enc_inputs = torch.zeros(0,0).long()
        self.dec_outputs = torch.zeros(0,0).long()
        self.dec_inputs = torch.zeros(0,0).long()
        self.demo_sentences = []
        self.hp = ModelHyperParams(hyper_params)

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

    def _update_dataset(self):
        src_idxs, tgt_idxs, dec_start = [], [], []
        for src_line, tgt_line in zip(self.sources, self.targets):
            x = [self.src_vocab.get(word, 1) for word in (src_line + ' ' + self.terminate_symbol).split()]   # 1: UNKNOWN
            y = [self.tgt_vocab.get(word, 1) for word in (tgt_line + ' ' + self.terminate_symbol).split()]
            y_s = [self.tgt_vocab.get(word, 1) for word in (self.start_symbol + ' ' + tgt_line).split()]
            if len(x) <= self.max_src_seq_len and len(y) <= self.max_tgt_seq_len:
                x.extend([0 for _ in range(self.max_src_seq_len - len(x))])
                y.extend([0 for _ in range(self.max_tgt_seq_len - len(y))])
                y_s.extend([0 for _ in range(self.max_tgt_seq_len - len(y_s))])
                src_idxs.append(x)
                tgt_idxs.append(y)
                dec_start.append(y_s)
        self.enc_inputs = torch.LongTensor(src_idxs)
        self.dec_outputs = torch.LongTensor(tgt_idxs)
        self.dec_inputs = torch.LongTensor(dec_start)

    # 数据加载函数，提供给子类调用
    def create_data(self, sources, targets):
        init_vocab = {self.pad_symbol: 0, self.unknown_symbol: 1, self.start_symbol: 2, self.terminate_symbol: 3}
        src_words = list(set(' '.join(sources).split()))   # set强制转换为list,是为方便排序，确保生成的idx是稳定的。
        tgt_words = list(set(' '.join(targets).split()))
        src_words.sort()
        tgt_words.sort()
        src_vocab = init_vocab.copy()
        src_vocab.update(  # i+4: 前面4个idx用于标志符号，文本单词的idx 向后移动 4位
            {w : i+4 for i, w in enumerate(src_words)})
        tgt_vocab = init_vocab.copy()
        tgt_vocab.update(
            {w : i+4 for i, w in enumerate(tgt_words)})
        src_idx2w = {v : k for k,v in src_vocab.items()}
        tgt_idx2w = {v: k for k, v in tgt_vocab.items()}

        src_idxs, tgt_idxs, dec_start = [], [], []
        for src_line, tgt_line in zip(sources, targets):
            x = [src_vocab.get(word, 1) for word in (src_line + ' ' + self.terminate_symbol).split()]   # 1: UNKNOWN
            y = [tgt_vocab.get(word, 1) for word in (tgt_line + ' ' + self.terminate_symbol).split()]
            y_s = [tgt_vocab.get(word, 1) for word in (self.start_symbol + ' ' + tgt_line).split()]
            if len(x) <= self.max_src_seq_len and len(y) <= self.max_tgt_seq_len:
                x.extend([0 for _ in range(self.max_src_seq_len - len(x))])
                y.extend([0 for _ in range(self.max_tgt_seq_len - len(y))])
                y_s.extend([0 for _ in range(self.max_tgt_seq_len - len(y_s))])
                src_idxs.append(x)
                tgt_idxs.append(y)
                dec_start.append(y_s)
        enc_inputs = torch.LongTensor(src_idxs)
        dec_outputs = torch.LongTensor(tgt_idxs)
        dec_inputs = torch.LongTensor(dec_start)

        self.sources = sources
        self.targets = targets
        self.enc_inputs = enc_inputs
        self.dec_outputs = dec_outputs
        self.dec_inputs = dec_inputs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_idx2w = src_idx2w
        self.tgt_idx2w = tgt_idx2w

    def preprocess(self, sentences):
        enc_inputs = []
        for sentence in sentences:
            enc_input = [self.src_vocab[w] for w in (sentence + ' ' + self.terminate_symbol).split()]
            enc_input.extend(0 for _ in range(self.max_src_seq_len - len(enc_input)))
            enc_inputs.append(enc_input)
        return torch.LongTensor(enc_inputs)

    def to_src_sentence(self, enc_input, sep=' '):
        return sep.join([self.src_idx2w[idx.item()] for idx in enc_input if idx > 0])

    def to_tgt_sentence(self, dec_output, sep=' ', first=False):
        if first:
            return sep.join([self.tgt_idx2w[idx.item()] for idx in dec_output if idx >0]).split(
                self.terminate_symbol)[0].strip()
        else:
            return sep.join([self.tgt_idx2w[idx.item()] for idx in dec_output if idx >0])

    def update_vocab(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_idx2w = {v : k for k,v in src_vocab.items()}
        self.tgt_idx2w = {v: k for k, v in tgt_vocab.items()}
        self._update_dataset()

    def update_hyper_params(self, hyper_params):
        self.hp.load(hyper_params)


class ModelHyperParams:
    def __init__(self, params=None):
        self.num_layers = 6            # number of encoder/decoder layers
        self.num_heads = 8             # number of heads for Multi-Head Attention block
        self.hidden_units = 512
        self.feed_forward_units = 512
        self.num_lstm_layers = 2
        if params:
            self.load(params)

    def load(self, params):
        self.num_layers = params.get('num_layers', self.num_layers)
        self.num_heads = params.get('num_heads', self.num_heads)
        self.hidden_units = params.get('hidden_units', self.hidden_units)
        self.feed_forward_units = params.get('feed_forward_units', self.feed_forward_units)
        self.num_lstm_layers = params.get('num_lstm_layers', self.num_lstm_layers)

    def text(self):
        params = {'num_layers': self.num_layers, 'num_heads': self.num_heads,
                  'hidden_units': self.hidden_units, 'feed_forward_units': self.feed_forward_units,
                  'num_lstm_layers': self.num_lstm_layers}
        return js.dumps(params, indent=2)

