import itertools

import pkuseg
import os
import brain.config as config
import json
from collections import defaultdict


output_path = "/Users/heathen/Desktop/train_data/data_analysis"
knowledge_path = os.path.join(os.path.dirname(__file__), 'KG_private/output')
knowledge_files = os.listdir(knowledge_path)
file_paths = [os.path.join(knowledge_path, f) for f in knowledge_files]

knowledge_dict = {}

for spo_path in file_paths:
    # print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
    with open(spo_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 拿到三元组
                subject, pred, objection, kind = line.strip().split("\t")
            except Exception as e:
                print(e)
                print("[KnowledgeGraph] Bad spo:", line)

            # 如果主语在查找表中
            if subject in knowledge_dict.keys():
                knowledge_dict[subject].append((pred, objection, kind))
            else:
                knowledge_dict[subject] = [(pred, objection, kind)]

vocab = list(knowledge_dict.keys()) + config.NEVER_SPLIT_TAG

tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=vocab)


def get_file_name_without_extension(file_path):
    # 获取文件的基本名称
    base_name = os.path.basename(file_path)
    # 分离文件件名和后缀
    file_name, _ = os.path.splitext(base_name)
    return file_name


def calculate_feature(file_path):
    father_name = os.path.basename(os.path.dirname(file_path))
    folder_path = os.path.join(output_path, father_name)
    os.makedirs(folder_path, exist_ok=True)
    answer = []
    output_file_name = get_file_name_without_extension(file_path) + "_analysis_second.txt"
    output_file_path = os.path.join(folder_path, output_file_name)

    print(f"正在处理{father_name}/{os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as f:
        total_different_entity = 0
        for num, single in enumerate(f):
            data = '[CLS]' + json.loads(single)['Prompt']
            # print("正在处理第{}句".format(num))
            word_list = tokenizer.cut(data[:256])
            news_dict = defaultdict(set)
            # 处理了知识所包含的目标实体，使其不再出现即使是不同类别但是同名的实体
            for word in word_list:
                if word in knowledge_dict:
                    for k in knowledge_dict[word]:
                        if k[1] not in news_dict:
                            news_dict[word].add((k[1], k[2]))

            different_entity, count_kind = count_diff_entity(knowledge_dict, answer, news_dict)
            total_different_entity += different_entity
            if different_entity != 0:
                answer.append(
                    f"第{num}句有{different_entity}处不同实体知识之间所含实体有关系, 有{count_kind}种不同的关系")

        if total_different_entity != 0:
            answer.append(f"总共有{total_different_entity}处不同实体知识之间所含实体有关系")

    print(f"{father_name}/{os.path.basename(file_path)}总共有{total_different_entity}处同句子不同实体知识重叠")
    if answer:
        with open(f"{output_file_path}", 'w') as f:
            f.write('\n'.join(answer))


def count_diff_entity(knowledge: dict, output: list, word_dict: dict):
    total_num = 0
    knowledge_keys = set()

    for i, j in itertools.combinations(word_dict.keys(), 2):
        if i in j or j in i:
            continue
        word_list1 = set(ex1[1] for ex1 in knowledge[i])
        word_list2 = set(ex2[1] for ex2 in knowledge[j])
        if i in word_list2 or j in word_list1:
            continue
        for ex1 in word_dict[i]:
            if ex1[0] not in knowledge:
                continue

            for k in knowledge[ex1[0]]:
                if k[1] in word_list2:
                    continue
                for ex2 in word_dict[j]:
                    if k[1] == ex2[0] and k[2] == ex2[1]:
                        key = (i, ex1[0], j, ex2[0], k[0])
                        if key not in knowledge_keys:
                            knowledge_keys.add(key)
                            output.append(f"实体 {i} 的 {ex1} 和 实体 {j} 的 {ex2}  有 {k[0]}  关系")
                        total_num += 1
                        break  # 找到匹配关系后跳出 ex2 循环，避免重复检查
    return total_num, len(knowledge_keys)


# calculate_feature("/Users/heathen/Desktop/git-now/K-BERT/output.jsonl")
data_path = "/Users/heathen/Desktop/train_data/data"
data_folders = [os.path.join(data_path, name) for name in os.listdir(data_path) if
                os.path.isdir(os.path.join(data_path, name))]
print(data_folders)
# data_folders = ['/Users/heathen/Desktop/train_data/data/lian_v1_article']
jsonl_files = []
for folder in data_folders:
    jsonl_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jsonl")])
# print(jsonl_files)
for file in jsonl_files:
    calculate_feature(file)
