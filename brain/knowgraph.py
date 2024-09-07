# coding: utf-8
"""
KnowledgeGraph
"""

import brain.config as config
import pkuseg
import numpy as np
from line_profiler_pycharm import profile


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict= self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    # 创建查找表--根据给定的spo文件
    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # 拿到三元组 # 已经变为四元组
                        # subject, relation, target, target_class =

                        words = line.strip().split("\t")
                        if len(words) == 3:
                            subject, relation, target = words
                        elif len(words) == 4:
                            subject, relation, target, target_class = words
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    # 如果有谓语
                    if self.predicate:
                        value = relation + target
                    else:
                        value = target
                    # 如果主语在查找表中
                    if subject in lookup_table.keys():
                        lookup_table[subject].add(value)
                    else:
                        lookup_table[subject] = {value}
        return lookup_table

    # max entities: 限制实体的数量么？
    """
    max_entities 参数的作用是限制每个词语可以关联的实体数量。
    这可以防止某个词语关联过多的实体，从而导致计算复杂度过高。-- 理解错误
    这里作出修改，可能不是这个原因，而是为了防止知识图谱中的实体数量过多，导致在原有句子长度的基础上，知识句子长度过长丢失信息
    """

    @profile
    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]  # 使用了pkuseg分词库的 分出来的各个句子啊
        # 例子：split_sent = [['[CLS]', '我', '是', '中国', '人']]
        # 要返回的四位主角
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []

            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                # entities是一个列表，用于存储与当前词语相关的实体 ，最多max_entities个
                sent_tree.append((token, entities))
                # [('CLS', [])（'我'，[]）, ('是',[]), ('中国',['首都北京'， '是城市']), ('人',[])]--sent_tree

                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    """[pos_idx + i for i in range(1, len(token) + 1)]为token中的每个字符生成一个从pos_idx + 1开始的连续位置索引列表，[
                    abs_idx + i for i in range(1, len(token) + 1)]为每个字符生成一个从abs_idx + 1开始的连续绝对索引列表"""
                    token_pos_idx = [pos_idx + i for i in range(1, len(token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(token) + 1)]

                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    # entities 是一个列表，用于存储与当前词语相关的实体。对于每个实体，将其每个字符作为一个单独的词语添加到知识句子中。
                    # 构建知识的位置索引列表和绝对索引列表
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)
                # 这里就是知识的拼接，将token和entities拼接起来 相对索引和绝对索引
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            # 第一个子循环结束

            # Get know_sent and pos
            # 这是一个列表，用于存储知识句子。知识句子是由原始句子中的词语和与这些词语相关的实体组成的。对于每个词语，
            # 如果它在特殊标记列表（self.special_tags）中，那么就直接将它添加到知识句子中；
            # 否则，将它的每个字符作为一个单独的词语添加到知识句子中。然后，对于与这个词语相关的每个实体，
            # 将实体的每个字符作为一个单独的词语添加到知识句子中。
            know_sent = []
            pos = []  # 用于存储每个词语在句子中的位置索引
            seg = []  # 用于存储每个词语的分段标记。对于每个词语，如果它在特殊标记列表中，那么其分段标记为0；
            # 否则，其分段标记为0。对于每个实体，其分段标记为1。
            # sent_tree.append((token, entities))  token是词语，entities是与词语相关的实体
            # 第二轮循环开始
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                # 拆词， 比如 '吕久鸿' 拆成 ['吕', '久', '鸿']， 放到know_sent中
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]  # 把[([0],[]),([1,2],[])...] 拼成[0,1,2...] (# soft position embedding)
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])  # 把知识先打散成列表，然后拼街到词的后面
                    know_sent += add_word
                    # 知识被标为1
                    seg += [1] * len(add_word)
                    # 將知识点的相对位置索引添加到词汇的后续位置中
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for now_id in src_ids:
                    # 对于主干句子中 所有的词语，以及“当前的实体”所属的全部分支
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[now_id, visible_abs_idx] = 1
                for ent in item[1]:  # item[1] : List[List[int]]
                    for now_id in ent:
                        # 这里其实就是对本分支加上分支所属的实体进行可视
                        visible_abs_idx = ent + src_ids  # ent是实体的绝对索引，src_ids是所属知识实体的绝对索引
                        visible_matrix[now_id, visible_abs_idx] = 1  # 这里 其实都是numpy ndarray，所以才能这么赋值
            # 填充与删除多余的部分
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                # 填充模式， 行-维度 在行后面填充pad_num行 0， 列-维度 在列后面填充pad_num列 0 常量模式填充
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
