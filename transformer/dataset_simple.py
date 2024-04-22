import codecs
from dataset import TransDataset

sentences = [['我 喜欢 猫 。', 'I like cats .'],
             ['我 不 喜欢 黑 猫 。', "I don't like black cat ."],
             ['我 喜欢 白 猫 。', 'I like white cat .'],
             ['我 喜欢 小 花猫 。', 'I like little tabby cat .'],
             ['我 有 一只 白 猫 。', 'I have a white cat .'],
             ['我 有 两只 小 花猫 。', 'I have two little tabby cat .'],
             ['我 没有 黑 猫 。', 'I have no black cat .'],
             ['我', 'I'],
             ['喜欢', 'like'],
             ['不', "don't"],
             ['小', 'little'],
             ['猫', 'cat'],
             ['花猫', 'tabby cat'],
             ['白', 'white'],
             ['黑', 'black'],
             ['有', 'have'],
             ['没有', 'have no'],
             ['一只', 'a'],
             ['两只', 'two'],
             ['。', '.'],
             ]

demo_sentences = ['我 喜欢 猫 。',
                  '我 喜欢 黑 猫 。',
                  '我 有 两只 小 花猫 。',
                  '我 有 两只 小 花猫',
                  '我 有 两只 白 猫 。',
                  '一只 小 花猫',
                  '一只 白 猫',
                  '我 有 一只 黑 猫',
                  '我 没有 白 猫',
                  '白 猫',
                  '小 花猫',
                  ]


# 定义语言翻译数据集的通用接口
class SimpleData(TransDataset):
    def __init__(self, data_type='train', max_lines=0):
        super(SimpleData, self).__init__()
        self.max_lines = max_lines
        self.max_src_seq_len = 10
        self.max_tgt_seq_len = 10

        self.load_data(data_type)
        self.demo_sentences = demo_sentences

    def load_data(self, data_type):
        sources = [lines[0].strip() for lines in sentences]
        targets = [lines[1].strip() for lines in sentences]
        if self.max_lines > 0:
            sources = sources[:self.max_lines]
            targets = targets[:self.max_lines]
        self.create_data(sources, targets)
