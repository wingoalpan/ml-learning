
import sys, os
import argparse
import json as js
import torch
import torch.utils.data as Data
from dataset_P3n9W31 import TP3n9W31Data
from dataset_simple import SimpleData
import bleu
from collections.abc import Iterable
import transformers.modeling_utils as modeling_utils

import transformer
import mt_lstm
import bench_test
from state import save_pretrained, save_as_safetensors
from transformer import Transformer, validate
from transformer_addon import merge_vocab, add_adapter, AdaptiveLinear

import wingoal_utils.common as CM
from wingoal_utils.common import (
    set_log_file,
    log,
    logs
)
sys.path.append('..\\utils')
import dl_utils

set_log_file(os.path.split(__file__)[-1], timestamp=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    last_states = dl_utils.get_last_state('simple_dropout0.3', 'lstm', 5)
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


def test_ngrams():
    candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    refs = ['It is a guide to action that ensures that the military will forever heed Party commands',
            'It is the guiding principle which guarantees the military forces always being under the command of the Party',
            'It is the practical guide for the army always to heed the directions of the party']

    refs_counter = bleu.ngrams(refs, 2)
    pred_counter = bleu.ngrams(candidate, 2)
    print('refs 2-grams:', refs_counter)
    print('pred 2-grams:', pred_counter)
    print('clip 2-grams:', pred_counter & refs_counter)


def test_bleu():
    candidate_list = ['It is a guide to action which ensures that the military always obeys the commands of the party',
                      'I wish the world keep peace and prosperous']
    refs_list = [['It is a guide to action that ensures that the military will forever heed Party commands',
                  'It is the guiding principle which guarantees the military forces always being under the command of the Party',
                  'It is the practical guide for the army always to heed the directions of the party'],
                 ['I wish the world is always peaceful and prosperous',
                  'I hope that the world is always peaceful and prosperous'
                 ]]

    print('bleu-1 score:', bleu.bleu(candidate_list, refs_list, 1))
    print('bleu-2 score:', bleu.bleu(candidate_list, refs_list, 2))
    print('bleu-3 score:', bleu.bleu(candidate_list, refs_list, 3))
    print('bleu-4 score:', bleu.bleu(candidate_list, refs_list, 4))


def test_load_json():
    node = {'name': '张三', '年龄': 16}
    CM.save_json(node, 'logs/test_save_json.json')

    obj = CM.load_json('logs/test_save_json.json')
    print(js.dumps(obj, indent=2, ensure_ascii=False))


def test_checkpoints():
    checkpoints = None
    if isinstance(checkpoints, list):
        checkpoints = iter(checkpoints)
    checkpoint = next(checkpoints) if isinstance(checkpoints, Iterable) else -1
    print(checkpoint)

    checkpoints = 3
    if isinstance(checkpoints, list):
        checkpoints = iter(checkpoints)
    checkpoint = next(checkpoints) if isinstance(checkpoints, Iterable) else -1
    print(checkpoint)

    checkpoints = [3, 4, 6]
    if isinstance(checkpoints, list):
        checkpoints = iter(checkpoints)
    checkpoint = next(checkpoints) if isinstance(checkpoints, Iterable) else -1
    print(checkpoint)


bench_test_sentences = [('我 喜欢 黑 猫 。',  'I like black cat .'),
                        ('我 有 一只 小 花猫 。',  'I have a little tabby cat . '),
                        ('我 不 喜欢 白 猫', "I don't like white cat"),
                        ('我 有 两只 黑 猫', 'I have two black cat'),
                        ('我 没有 白 猫', 'I have no white cat'),
                        ('我 喜欢 小 花猫', 'I like little tabby cat'),
                        ('我 喜欢 小 黑 猫', 'I like little black cat'),
                        ('一只 黑 猫', 'a black cat'),
                        ('两只 白 猫', 'two white cat'),
                        ('没有 两只 白 猫', 'have no two white cat'),
                        ('不 喜欢 小 花猫', "don't like little tabby cat"),
                        ('喜欢 小 白 猫', "like little white cat"),
                        ]


def test_train():
    corpora = SimpleData()
    batch_size = 6
    dropout = 0.
    checkpoint_interval = 0
    checkpoints = [5,10,20,40,75,100]
    num_epochs = 1000
    loader = Data.DataLoader(corpora, batch_size, True)
    model = transformer.create_model(corpora, name='lr_test_drop0.1', dropout=dropout).to(device)
    transformer.train(model, loader, num_epochs, force_retrain=False, checkpoint_interval=100, checkpoints=None)

    test_sentences = bench_test_sentences
    src_sentences = [item[0] for item in test_sentences]
    refs_sentences = [[item[1]] for item in test_sentences]
    enc_inputs = corpora.preprocess(src_sentences).to(device)

    dec_outputs = model.translate(enc_inputs)
    candidates = []
    for enc_input, dec_output, refs in zip(enc_inputs, dec_outputs, refs_sentences):
        prediction = corpora.to_tgt_sentence(dec_output, first=True)
        print(js.dumps({
            'source': corpora.to_src_sentence(enc_input),
            'translated': prediction,
            'references': refs if len(refs) > 1 else refs[0]
        }, indent=2, ensure_ascii=False))
        candidates.append(prediction)
    metric_bleu = bleu.bleu(candidates, refs_sentences, 2)
    log('translation bleu:', metric_bleu)


def test_model_trained():
    epochs = [50, 200]
    print(bench_test._model_trained('simple_dropout0.0', 'transformer', epochs))
    epochs = [50, 100, 200, 300, 400, 600, 800, 1000]
    print(bench_test._model_trained('simple_dropout0.0', 'transformer', epochs))
    epochs = [50, 200, 500]
    print(bench_test._model_trained('simple_dropout0.0', 'transformer', epochs))
    test_cases = CM.load_json('benchmark/test-benchmark.json')
    test_cases = bench_test.benchmark_cases if test_cases is None else test_cases
    log('training transformer... ')
    bench_test.train(test_cases, transformer, force_retrain=False)
    log('training mt_lstm... ')
    bench_test.train(test_cases, mt_lstm, force_retrain=False)
    CM.save_json(test_cases, 'benchmark/test-benchmark-test.json')


def test_save_safetensors():
    state_file = 'state_dict/transformer-classical-800-model.pkl'
    model_save_dir = 'models/transformer-P3n9W31_0.1-800'
    save_as_safetensors(state_file, save_directory=model_save_dir, max_shard_size='60MB')


def test_save_pretrained():
    state_file = 'state_dict/transformer-classical-600-model.pkl'
    model_save_dir = 'models/transformer-P3n9W31_0.1-600'
    corpora, name, batch_size, num_epochs = TP3n9W31Data(), 'classical', 48, 800
    _dropout = 0.1
    model = Transformer(corpora, name, dropout=_dropout).to(device)
    model.load_state_dict(torch.load(state_file))
    save_pretrained(model, save_directory=model_save_dir, max_shard_size='100MB')


def test_load_safetensors():
    model_save_dir = 'models/transformer-P3n9W31_0.1-600'

    corpora, name, batch_size, num_epochs = TP3n9W31Data(), 'classical', 48, 800
    _dropout = 0.1
    model = Transformer(corpora, name, dropout=_dropout).to(device)
    modeling_utils.load_sharded_checkpoint(model, model_save_dir, strict=True, prefer_safe=True)
    model.eval()

    log('validating model ...')
    result = validate(model, 100)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))


def test_merge_vocab():
    model_save_dir = 'models/transformer-P3n9W31_0.1-800'

    corpora, name = TP3n9W31Data(), 'classical'
    _dropout = 0.1
    model = Transformer(corpora, name, dropout=_dropout).to(device)
    modeling_utils.load_sharded_checkpoint(model, model_save_dir, strict=True, prefer_safe=True)

    inc_copora = SimpleData()
    merge_vocab(model, inc_copora)
    model.train()

    batch_size, num_epochs = 6, 200
    inc_copora.update_vocab(corpora.src_vocab, corpora.tgt_vocab)
    loader = Data.DataLoader(inc_copora, batch_size, True)
    transformer.train(model, loader, num_epochs, force_retrain=True, checkpoint_interval=-1)

    model.eval()

    log('validating model ...')
    result = validate(model, 100)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))

    test_sentences = [item[0] for item in bench_test.benchmark_test_sentences]
    # test_sentences = inc_copora.sources
    enc_inputs = model.corpora.preprocess(test_sentences)

    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子 翻译成英文句子: ")
    dec_outputs = transformer.predict_batch(model, enc_inputs)
    i = 0
    for enc_input, pred in zip(enc_inputs, dec_outputs):
        print('%s.' % (i + 1), corpora.to_src_sentence(enc_input, ''))
        print('->', corpora.to_tgt_sentence(pred.squeeze(), first=True), '\n')
        i += 1


def test_add_adapter():
    model_save_dir = 'models/transformer-P3n9W31_0.1-800'

    corpora, name = TP3n9W31Data(), 'classical'
    _dropout = 0.1
    model = Transformer(corpora, name, dropout=_dropout).to(device)
    modeling_utils.load_sharded_checkpoint(model, model_save_dir, strict=True, prefer_safe=True)

    inc_copora = SimpleData()
    merge_vocab(model, inc_copora)
    add_adapter(model, rank=8)

    param_count, net_info = dl_utils.get_net_detail(model, show_param_shape=True)
    logs('model detail:', net_info)
    log('total parameters: ', param_count)

    model.train()

    batch_size, num_epochs = 10, 500
    inc_copora.update_vocab(corpora.src_vocab, corpora.tgt_vocab)
    loader = Data.DataLoader(inc_copora, batch_size, True)
    transformer.train(model, loader, num_epochs, force_retrain=True, checkpoint_interval=-1)

    model.eval()

    AdaptiveLinear.use_adapter = False
    log('validating model ...')
    result = validate(model, 100)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))

    AdaptiveLinear.use_adapter = True
    test_sentences = [item[0] for item in bench_test.benchmark_test_sentences]
    enc_inputs = model.corpora.preprocess(test_sentences)

    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子 翻译成英文句子: ")
    dec_outputs = model.translate(enc_inputs)
    i = 0
    for enc_input, pred in zip(enc_inputs, dec_outputs):
        print('%s.' % (i + 1), corpora.to_src_sentence(enc_input, ''))
        print('->', corpora.to_tgt_sentence(pred.squeeze(), first=True), '\n')
        i += 1


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

