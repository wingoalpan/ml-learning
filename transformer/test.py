
import sys, os
import argparse
import json as js
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


def main():
    test_data_train()
    test_data_test()
    test_simple_data()
    test_simple_data_2()


def _show_data(data):
    print('sources len:', len(data.sources))
    print('targets len:', len(data.targets))
    print('source vocabs len:', len(data.src_vocab))
    print('target vocabs len:', len(data.tgt_vocab))
    print('source vocabs size:', data.src_vocab_size)
    print('target vocabs size:', data.tgt_vocab_size)
    print('hyperparams:', js.dumps(data.hyperparams, indent=2))


def test_data_train():
    data = TP3n9W31Data('train')
    _show_data(data)


def test_data_test():
    data = TP3n9W31Data('test')
    _show_data(data)


def test_simple_data():
    data = SimpleData('train')
    _show_data(data)


def test_simple_data_2():
    data = SimpleData('test')
    _show_data(data)


def test_data_detail():
    data = SimpleData('train')
    print('sources: ', js.dumps(data.sources, ensure_ascii=False))
    print('targets: ', js.dumps(data.targets))
    print('source vocab: ', js.dumps(data.src_vocab, ensure_ascii=False, indent=2))
    print('target vocab: ', js.dumps(data.tgt_vocab, indent=2))
    print('enc_inputs: ', data.enc_inputs)
    print('dec_outputs: ', data.dec_outputs)


def test_data_max():
    data = TP3n9W31Data('train', max_lines=10)
    print('sources: ', js.dumps(data.sources, indent=2, ensure_ascii=False))
    print('targets: ', js.dumps(data.targets, indent=2))
    print('source vocab: ', js.dumps(data.src_vocab, ensure_ascii=False, indent=2))
    print('target vocab: ', js.dumps(data.tgt_vocab, indent=2))


def test_preprocess():
    data = SimpleData('train')
    sentences = data.demo_sentences
    enc_inputs = data.preprocess(sentences)

    print('sources: ', js.dumps(sentences, ensure_ascii=False))
    print('enc_inputs: ', enc_inputs)
    print('source vocab: ', js.dumps(data.src_vocab, ensure_ascii=False, indent=2))


def test_last_state():
    last_states = dl_utils.get_last_state('simple')
    print(js.dumps(last_states, indent=2))


def test_trans():
    with open('logs/translation_test-1.json', 'r', encoding='utf8') as f:
        lines = js.load(f)
    for rec in lines[:10]:
        print(rec['source'])


def test_state():
    print(dl_utils.get_model_state('simple', 'transformer'))


def test_set():
    l = ['zhang', 'san', 'li', 'si', 'wang', 'zhang', 'jiang', 'si', 'ren', 'bang']
    ls = list(set(l))
    ls.sort()
    for i, n in enumerate(ls):
        print(i, ':', n)


def test_hp():
    corpora = SimpleData()
    print('hyper params: \n', corpora.hp.text())
    corpora.update_hyper_params({'num_layers': 4})
    print('new hyper params: \n', corpora.hp.text())
    corpora.update_hyper_params({'hidden_units': 256})
    print('latest hyper params: \n', corpora.hp.text())


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

