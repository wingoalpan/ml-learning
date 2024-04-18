
import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import zipfile
import re
import json as js

sys.path.append('..\\..\\wingoal_utils')
from common import (
    log,
    logs
)


# 检查一个 nn.Module是否有子模块
# 输入: module -- 必须是一个 nn.Module类型的实例
# 输出: bool
#          True  -- module有包含子模块
#          False -- module没有子模块
def _has_children(module):
    for _ in module.children():
        return True
    return False


def check_net(net, x, net_name='-', level=0, as_singles=None):
    lines = []
    if level == 0:
        lines.append('input data shape: {}'.format(x.shape))
    indent_str = ' ' * level * 2
    if as_singles is None:
        as_singles = []

    class_name = net.__class__.__name__
    if class_name in as_singles:
        lines.append('%s(%s): %s' % (indent_str, net_name, net.__class__.__name__))
        x = net(x)
        lines.append('{indent}{output_flag} {shape}'.format(indent=' ' * 10, output_flag='=' * 4, shape=x.shape))
    elif not _has_children(net):
        lines.append('%s(%s): %s' % (indent_str, net_name, net))
        x = net(x)
        lines.append('{indent}{output_flag} {shape}'.format(indent=' ' * 10, output_flag='=' * 4, shape=x.shape))
    else:
        children_lines = []
        for name, child in net.named_children():
            x, child_info = check_net(child, x, name, level + 1, as_singles)
            children_lines.append(child_info)
        lines.append('%s(%s): %s (' % (indent_str, net_name, net.__class__.__name__))
        lines.extend(children_lines)
        lines.append('%s)' % indent_str)

    if level == 0:
        lines.append('output data shape: {}'.format(x.shape))
    return x, '\n'.join(lines)


def get_net_detail(net, net_name='-', show_param_shape=False, level=0):
    lines = []
    param_count = 0
    indent_str = ' ' * level * 2
    if not _has_children(net):
        param_count, _ = _count_params(net, show_param_shape=False)
        if param_count > 0:
            lines.append('%s(%s): %s [params: %s]' % (indent_str, net_name, net, param_count))
            if show_param_shape:
                _, params_shape = _count_params(net, show_param_shape=True)
                lines.append(params_shape)
        else:
            lines.append('%s(%s): %s' % (indent_str, net_name, net))
    else:
        children_lines = []
        for name, child in net.named_children():
            p_count, child_info = get_net_detail(child, name, show_param_shape, level + 1)
            param_count += p_count
            children_lines.append(child_info)
        lines.append('%s(%s): %s [params: %s] (' % (indent_str, net_name, net.__class__.__name__, param_count))
        lines.extend(children_lines)
        lines.append('%s)' % indent_str)
    return param_count, '\n'.join(lines)


def _count_params(net, show_param_shape=False):
    p_count = 0
    lines = []
    for p in net.parameters():
        shape = p.data.shape
        if show_param_shape:
            lines.append('{indent}{param_flag} {shape}'.format(indent=' ' * 10, param_flag='#' * 4, shape=shape))
        if len(shape) > 1:
            cnt = shape[0]
            for i in range(1, len(shape)):
                cnt = cnt * shape[i]
            p_count += cnt
        else:
            p_count += shape[0]
    return p_count, '\n'.join(lines)


def get_last_state(prefix='', model_type='', max_epoch=0, state_dir='state_dict'):
    prefix = prefix if prefix else 'default'
    states = {}
    pat = r'(?P<last_epoch>\d*)-model.pkl'
    pat = '-'.join([p for p in [model_type, prefix, pat] if p])

    file_names = os.listdir(state_dir)
    for file_name in file_names:
        m = re.match(pat, file_name)
        if not m:
            continue
        last_epoch = int(m.group('last_epoch'))
        if (max_epoch > 0) and last_epoch > max_epoch:
            continue

        if states:
            if last_epoch >= states['last_epoch']:
                states = {'last_epoch': last_epoch, 'file_name': os.path.join(state_dir, file_name)}
        else:
            states = {'last_epoch': last_epoch, 'file_name': os.path.join(state_dir, file_name)}
    return states


def get_model_state(name='', model_type='', model_dir='models'):
    name = name if name else 'default'
    if not os.path.exists(os.path.join(model_dir, 'model-list.json')):
        return {}

    with open(os.path.join(model_dir, 'model-list.json'), 'r', encoding='utf8') as f:
        model_info = js.load(f)

    key = '-'.join([p for p in [model_type, name] if p])
    if key not in model_info.keys():
        return {}

    last_epoch = model_info[key]['last_epoch']
    file_name = model_info[key]['state_file']
    return {'last_epoch': last_epoch, 'file_name': os.path.join(model_dir, file_name)}


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals is not None and y2_vals is not None:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        # plt.legend(legend)
    plt.show()


def scatter(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x_vals, y_vals)
    if x2_vals is not None and y2_vals is not None:
        plt.scatter(x2_vals, y2_vals, linestyle=':')
        # plt.legend(legend)
    plt.show()


def cross_entropy_loss(y_hat, y):
    loss = - torch.log(y_hat.gather(1, y.view(-1, 1)))
    return loss.mean()


def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))


def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob
    

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def load_mnist(batch_size, validation = False):
    validate_loader = None
    if validation:
        train_db = datasets.MNIST('../data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])),
        train_db, validate_db = torch.utils.data.random_split(train_db, [50000,10000])
        validate_loader = torch.utils.data.DataLoader(validate_db, batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validate_loader


def load_fashion_mnist(batch_size, validation=False, resize=None):
    validate_loader = None
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)
    if validation:
        train_db = datasets.FashionMNIST('../data', train=True, download=True,
                                  transform=transform),
        train_db, validate_db = torch.utils.data.random_split(train_db, [50000,10000])
        validate_loader = torch.utils.data.DataLoader(validate_db, batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                           transform=transform),
            batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validate_loader


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_mnist_labels(labels):
    text_labels = ['0', '1', '2', '3', '4', '5',
                   '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    net.eval()  # 评估模式, 这会关闭dropout
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    net.train()  # 改回训练模式
    return acc_sum / n


# loss 用函数应为可累计的，例如 nn.CrossEntropyLoss(reduction="none") 或 nn.CrossEntropyLoss(reduction="sum")
# 不能够用 nn.CrossEntropyLoss(reduction="mean")
def evaluate_accuracy_loss(data_iter, net, loss):
    acc_sum, loss_sum, n = 0.0, 0.0, 0
    net.eval()  # 评估模式, 这会关闭dropout
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            loss_sum += loss(y_hat, y).sum().item()
            n += y.shape[0]
    net.train()  # 改回训练模式
    return acc_sum / n, loss_sum / n


def train(net, train_iter, test_iter, num_epochs, lr, device):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        net.train()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            batch_count += 1
        train_acc = evaluate_accuracy(train_iter, net)
        test_acc = evaluate_accuracy(test_iter, net)
        log('epoch %d, loss %.4f, accuracy (%.3f, %.3f), time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc, test_acc, time.time() - start))


def train_batch(net, X, y, loss, optimizer, device):
    if isinstance(X, list):
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    net.train()
    optimizer.zero_grad()
    y_hat = net(X)
    l = loss(y_hat, y)
    l.backward()
    optimizer.step()
    train_loss = l.item()
    train_acc = accuracy(y_hat, y)
    return train_loss, train_acc


def train_plus(net, train_iter, test_iter, loss, optimizer, device, num_epochs=10):
    net = net.to(device)
    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        for X, y in train_iter:
            train_batch(net, X, y, loss, optimizer, device)
        train_acc, train_loss = evaluate_accuracy_loss(train_iter, net, loss)
        test_acc, test_loss = evaluate_accuracy_loss(test_iter, net, loss)
        log('epoch %d, loss (%.6f, %.6f), accuracy (%.3f, %.3f), time %.1f sec'
              % (epoch + 1, train_loss, test_loss, train_acc, test_acc, time.time() - start))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def load_data_jaychou_lyrics():
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:10000]

    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    num_samples = (len(corpus_indices) -1) // num_steps
    epoch_size = num_samples // batch_size
    sample_indices = list(range(num_samples))
    random.shuffle(sample_indices)

    def _data(start_pos):
        return corpus_indices[start_pos: start_pos + num_steps]

    for i in range(epoch_size):
        pos = i * batch_size
        batch_indices = sample_indices[pos: pos + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    batch_len = len(corpus_indices) // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        pos = i * num_steps
        X = indices[:, pos: pos + num_steps]
        Y = indices[:, pos + 1: pos + 1 + num_steps]
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def load_airfoil_data():
    data = np.genfromtxt('../data/airfoil_self_noise.dat')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), torch.tensor(data[:1500, -1], dtype=torch.float32)


