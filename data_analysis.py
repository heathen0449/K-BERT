import pkuseg
import os
import brain.config as config
import json

knowledge_path = os.path.join(os.path.dirname(__file__), 'KG_private/output')
knowledge_files = os.listdir(knowledge_path)
file_paths = [os.path.join(knowledge_path, f) for f in knowledge_files]

knowledge_dict = {}

for spo_path in file_paths:
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

output_path = "/Users/heathen/Desktop/train_data/data_analysis"



def get_file_name_without_extension(file_path):
    # 获取文件的基本名称
    base_name = os.path.basename(file_path)
    # 分离文件件名和后缀
    file_name, _ = os.path.splitext(base_name)
    return file_name


def calculate_feature(file_path):
    father_name = os.path.basename(os.path.dirname(file_path))
    folder_path = os.path.join(output_path, father_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    answer = []
    output_file_name = get_file_name_without_extension(file_path) + "_analysis.txt"
    output_file_path = os.path.join(folder_path, output_file_name)
    print(f"正在处理{father_name}/{os.path.basename(file_path)}")
    with (open(file_path, 'r', encoding='utf-8') as f):
        total_same = 0
        for num, single in enumerate(f):
            data = '[CLS]' + json.loads(single)['Prompt']

            word_list = tokenizer.cut(data[:256])
            same = 0
            knowledge_set = []
            answer_set = set()
            for word in word_list:
                if word in knowledge_dict.keys():
                    now_word_set = set(pp[1] for pp in knowledge_dict[word])
                    for k in knowledge_set:
                        k_word_set = set(pp[1] for pp in k[1])

                        # 找到共有的对象
                        common_objects = now_word_set.intersection(k_word_set)

                        # 两个知识如果存在相同的对象，加一
                        for i in k[1]:
                            for j in knowledge_dict[word]:
                                if j[1] == i[1] and j[2] == i[2]:
                                    if word != k[0] and  word  not in k[0] and k[0] not in word:
                                        same += 1
                                        key1 = (k[0], i[0], i[1], word, j[0], j[1]),
                                        key2 = (word, j[0], j[1], k[0], i[0], i[1])
                                        if key1 not in answer_set and \
                                                key2 not in answer_set:
                                            answer_set.add(key1)
                                            answer.append(f"关系1 {k[0]} {i[0]} {i[1]}  关系2 {word} {j[0]} {j[1]}")
                    knowledge_set.append([word, knowledge_dict[word]])

            total_same += same
            if same != 0:
                answer.append(f"第{num}句有{same}处知识重叠， 有{len(answer_set)}种不同的知识重叠")

        if total_same != 0:
            answer.append(f"总共有{total_same}处知识重叠")

    print(f"{father_name}/{os.path.basename(file_path)}总共有{total_same}处知识重叠")
    if answer:
        with open(f"{output_file_path}", 'w') as f:
            for a in answer:
                f.write(a)
                f.write('\n')


data_path = "/Users/heathen/Desktop/train_data/data"
data_folders = [os.path.join(data_path, name) for name in os.listdir(data_path) if
                os.path.isdir(os.path.join(data_path, name))]
# print(data_folders)
# data_folders = ['/Users/heathen/Desktop/train_data/data/lian_v1_article']
jsonl_files = []
for folder in data_folders:
    jsonl_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jsonl")])
# print(jsonl_files)
for file in jsonl_files:
    calculate_feature(file)
