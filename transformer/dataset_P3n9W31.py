import os
import codecs
from dataset import TransDataset

source_train = "cn.txt"
target_train = "en.txt"
source_test = "cn.test.txt"
target_test = "en.test.txt"

demo_sentences = ['麦凯恩 担任 国会 议员 多 年 , 参加 过 越战 , 并 度过 了 5 年 半 的 战俘 生涯 。',
                  '林村乡 历史 悠久 , 居民 的 先祖 于 宋朝 末年 已 迁徙 至 此 , 至今 已 有 700 多 年 。',
                  '第一 , 进行 结构 调整 最 直接 的 目标 , 是 尽快 使 中国 经济 回复 到 正常 增长 的 轨道 上 去 。',
                  '第二 , 进行 结构 调整 最 直接 的 目标 , 是 尽快 使 中国 经济 回复 到 正常 增长 的 轨道 上 去 。',
                  '第一 , 进行 结构 调整 最 直接 的 目标 , 是 尽快 使 经济 回复 到 正常 增长 的 轨道 上 去 。',
                  '第一 , 进行 结构 调整 的 目标 , 是 尽快 使 经济 回复 到 正常 增长 的 轨道 上 去 。',
                  '第二 , 进行 结构 调整 最 直接 的 目标 , 是 尽快 使 经济 回复 到 正常 增长 的 轨道 上 。',
                  '每 天 晚上 都 可以 看 电视 , 除了 新闻 , 周末 还 可以 看看 娱乐 节目',
                  '布什 是 与 到访 的 中国 国务院 副总理 钱其琛 会谈 前 共同 与 媒体 记者 见面 时 作 此 表示 的 。',
                  '思想 政治 工作 是 经济 工作 和 其他 一切 工作 的 生命线 , 也 是 我军 的 优良 传统 和 重要 政治 优势 。',
                  '我们 有 工资 成本 较 低 、 素质 较 高 的 劳动力 , 将 会 极 大 地 释放 市场 拓展 潜力 。',
                  '到 去年 底 , 台商 在 祖国 大陆 投资 合同 金额 达 四百九十七亿 美元 , 实际 到位 金额 二百七十二亿 美元 。',
                  '三 个 代表 ” 重要 思想 是 中国 共产党 人 的 眼界 、 胸襟 以及 历史 责任心 的 理论 再现 。',
                  '博什格拉夫 没有 说 但 旁观者清 就 是 因为 美国 以为 自己 无所不在 无所不能 要求 所有 国家 唯它 的 马首是瞻',
                  ]

hyperparams = {
    'batch_size': 48,
    'hidden_units': 512,
    'feed_forward_units': 512,
    'num_heads': 8,
    'num_layers': 6,  # number of encoder/decoder layers
    'num_epochs': 200,
    'dropout': 0.4,    # dropout rate
    'lr': 1e-3,        # optimizer 's lr parameter
    'momentum': 0.99,  # optimizer 's momentum parameter
    'num_validate': 10,
    'model_name': '',  # 作为保存模型参数状态的文件名前缀
}


# 定义语言翻译数据集的通用接口
class TP3n9W31Data(TransDataset):
    def __init__(self, data_type='train', max_lines=0, corpora_dir='corpora'):
        super(TP3n9W31Data, self).__init__()
        self.max_lines = max_lines
        self.corpora_dir = corpora_dir
        self.load_data(data_type)
        self.demo_sentences = demo_sentences
        self.hyperparams = hyperparams

    def load_data(self, data_type):
        assert data_type in ["train", "test"]
        if data_type == "train":
            src_file, tgt_file = source_train, target_train
        else:
            src_file, tgt_file = source_test, target_test
        sources = [line.strip()
                   for line in codecs.open(os.path.join(self.corpora_dir, src_file), "r", "utf-8").read().split("\n")]
        targets = [line.strip()
                   for line in codecs.open(os.path.join(self.corpora_dir, tgt_file), "r", "utf-8").read().split("\n")]
        if self.max_lines > 0:
            sources = sources[:self.max_lines]
            targets = targets[:self.max_lines]
        self.create_data(sources, targets)
