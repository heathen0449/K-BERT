import torch.nn as nn
import torch
import argparse
import json
import os
from uer.utils.constants import *
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.model_saver import save_model
from brain import KnowledgeGraph
from uer.encoders.bert_encoder import BertEncoder
from uer.layers.embeddings import BertEmbedding
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score, recall_score
import numpy as np
# 分類結果只有兩類 0-單主題 1-多主題（待定）

class MyClassifier(nn.Module):
    """
        MyClassifier
        类使用了一个
        BERT
        模型来进行二分类任务。最后输出的
        prob
        是一个包含两个元素的张量，表示输入样本属于两个类别的概率。具体来说，prob
        的形状是[batch_size, 2]，其中
        batch_size
        是输入样本的数量。

        如果您希望输出的是用
        0 / 1
        表示的类别标签，而不是概率，您需要对
        prob
        进行进一步处理，通常是应用一个
        softmax
        函数并选择概率最高的类别作为预测结果。
    """

    def __init__(self, args):
        super(MyClassifier, self).__init__()
        self.embedding = BertEmbedding(args, len(args.vocab))
        self.encoder = BertEncoder(args)
        self.input_linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_linear = nn.Linear(args.hidden_size, 2)
        self.dropout = nn.Dropout(args.dropout)
        self.use_input_linear = args.use_input_linear

    def forward(self, src, mask, pos, vm):

        if self.use_input_linear:
            vm = self.input_linear(vm)
        emb = self.embedding(src, mask, pos)
        # 此时需要修改，在引入公共关系的情况下对其可以进行输入训练
        output = self.encoder(emb, mask, vm)
        # 提取cls 位置输出 用于分类
        # dropout 用于防止过拟合
        cls_output = self.dropout(output[:, 0, :])
        prob = self.output_linear(cls_output)
        return prob


#     def predict(self, src, mask, pos, vm):
#         prob = self.forward(src, mask, pos, vm)
#         prob = F.softmax(prob, dim=-1)  # 应用 softmax 函数
#         pred_labels = torch.argmax(prob, dim=-1)  # 获取概率最高的类别
#         return pred_labels
# 在 predict 方法中，我们首先调用 forward 方法获取 prob，然后应用 softmax 函数将其转换为概率分布，
# 最后使用 torch.argmax 获取概率最高的类别作为预测结果。这样，predict 方法返回的是用 0/1 表示的类别标签。
#
# 请注意，在训练过程中，您通常会使用 prob 进行损失计算，而不是直接使用类别标签。只有在推理阶段，您才需要将概率转换为类别标签。


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--vocab_path", default="./models/vocab.txt", type=str
                    , help="词汇表位置")
parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                    help="模型参数位置")
parser.add_argument("--pretrained_model_path", default="./models/google_model.bin", type=str,
                    help="预训练模型参数位置")
parser.add_argument("--batch_size", default=32, type=int, help="批量大小")
parser.add_argument("--seq_length", default=256, type=int, help="序列长度")
parser.add_argument("--output_model_path", default=f"./models/classifier_model.bin", type=str,
                    help="输出模型位置")
# Training options.
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout. 自设的默认的丢弃率")
parser.add_argument("--epochs_num", type=int, default=5,
                    help="Number of epochs. 训练轮数")
parser.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Learning rate.")
parser.add_argument("--use_input_linear", type=bool, default=False, help="是否使用输入线性层")

args  = parser.parse_args()
args = load_hyperparam(args)

vocab = Vocab()
vocab.load(args.vocab_path)
args.vocab = vocab
spo_files = []
spo_dir = os.path.join(os.getcwd(), 'KG_private/output')
for file_name in os.listdir(spo_dir):
    if file_name.endswith('.spo'):
        spo_files.append(os.path.join(spo_dir, file_name))

kg = KnowledgeGraph(spo_files=spo_files)
# 划分数据集
# 假设你的数据集是data和labels
# data = [...]  # 你的数据
# labels = [...]  # 你的标签
data = []
labels = []
data_dir = os.path.join(os.getcwd(), 'datasets/financial_news_data/Downloads')
for f in os.listdir(data_dir):
    if f.endswith('.jsonl'):
        with open(os.path.join(data_dir, f), 'r') as file:
            for line in file:
                line_file = json.loads(line)
                if line_file['Result'] == 'Unknown':
                    continue
                data.append((CLS_TOKEN + line_file['News'])[:256])
                if line_file['Result'] == 'False':
                    labels.append(0)
                else:
                    labels.append(1)


# 第一次划分：将数据分为训练集和临时集（临时集包括验证集和测试集）
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, train_size=0.7, random_state=42)

# 第二次划分：将临时集分为验证集和测试集
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

# train_datas , train_labels. val_datas, val_labels, test_datas, test_labels


model = MyClassifier(args)
model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)

# For simplicity, we use DataParallel wrapper to use multiple GPUs.
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

if torch.cuda.device_count() > 1:
    print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

model = model.to(device)


class CustomDataset(Dataset):
    def __init__(self, input_data, input_labels):
        self.data = input_data
        self.labels = input_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = CustomDataset(train_data, train_labels)
val_dataset = CustomDataset(val_data, val_labels)
test_dataset = CustomDataset(test_data, test_labels)

# 定义 DataLoader
# batch_size = 32
# 创建 DataLoader 对象
batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型、损失函数和优化器
# model = MyClassifier(args)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def process_batch(some_batch):
    news, label_list = some_batch
    know_sent, position, vm, seg = kg.add_knowledge_with_vm(news, add_pad=True, max_length=args.seq_length)

    # 确保 vm 张量的形状正确
    for i in range(len(vm)):
        vm[i] = vm[i].astype('bool')

    tokens_batch = [[vocab.get(t) for t in i] for i in know_sent]
    mask = []
    seg_tag = 1
    for i in know_sent:
        masking = []
        for j in i:
            if j == PAD_TOKEN:
                masking.append(0)
            else:
                masking.append(seg_tag)
            if j == SEP_TOKEN:
                seg_tag += 1
        mask.append(masking)

    # 确保所有张量都在同一设备上
    vm = np.array(vm)
    tokens_batch = torch.LongTensor(tokens_batch).to(device)
    mask = torch.LongTensor(mask).to(device)
    position = torch.LongTensor(position).to(device)
    vm_batch = torch.BoolTensor(vm).to(device) # 使用 BoolTensor 而不是 LongTensor
    labels = torch.LongTensor(label_list).to(device)

    # 打印张量的形状
    # print(f"tokens_batch shape: {tokens_batch.shape}")
    # print(f"mask shape: {mask.shape}")
    # print(f"position shape: {position.shape}")
    # print(f"vm_batch shape: {vm_batch.shape}")
    # print(f"labels shape: {labels.shape}")

    return tokens_batch, mask, position, vm_batch, labels


# 训练和验证循环
for epoch in range(args.epochs_num):
    print(f"现在是第{epoch+1}轮")
    # 训练阶段
    model.train()
    for batch in train_loader:
        know_sent_batch, mask_batch, position_batch, vm_batch, labels = process_batch(batch)
        optimizer.zero_grad()
        outputs = model(know_sent_batch, mask_batch, position_batch, vm_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            know_sent_batch, mask_batch, position_batch, vm_batch, labels = process_batch(batch)
            outputs = model(know_sent_batch, mask_batch, position_batch, vm_batch)
            #这行代码计算模型的损失。criterion是损失函数，例如交叉熵损失（nn.CrossEntropyLoss）。outputs是模型的输出（通常是未归一化的概率分布），labels是实际的标签。
            # 计算出的loss是一个张量，表示当前批次的损失值。我们将这个值加到val_loss中，以便在验证阶段结束时计算平均损失。
            loss = criterion(outputs, labels)
            #这行代码将当前批次的损失值累加到val_loss中。loss.item()将损失张量转换为Python的标量值，以便进行累加。
            # val_loss通常在循环外部初始化为0，用于记录整个验证集的总损失。
            val_loss += loss.item()
            # 这行代码从模型的输出中获取预测的类别。torch.max(outputs.data, 1)返回每行的最大值及其索引（即预测的类别）。
            # _表示忽略最大值，只保留索引（即predicted），这些索引代表模型预测的类别。
            _, predicted = torch.max(outputs.data, 1)
            # 这行代码累加当前批次的样本数量到total中。labels.size(0)返回当前批次的样本数量。
            # total通常在循环外部初始化为0，用于记录整个验证集的样本总数。
            total += labels.size(0)
            # 这行代码计算当前批次中预测正确的样本数量，并累加到correct中。predicted == labels生成一个布尔张量，表示每个预测是否正确，.sum()计算预测正确的样本数量，.item()将结果转换为Python标量。correct通常在循环外部初始化为0，用于记录整个验证集的正确预测总数。
            correct += (predicted == labels).sum().item()
            # 这行代码将当前批次的实际标签添加到all_labels列表中。labels.cpu().numpy()将标签张量移动到CPU并转换为NumPy数组，然后使用extend方法将其添加到列表中。
            # all_labels通常在循环外部初始化为空列表，用于记录整个验证集的所有实际标签。
            all_labels.extend(labels.cpu().numpy())
            # 这行代码将当前批次的预测结果添加到all_predictions列表中。predicted.cpu().numpy()将预测张量移动到CPU并转换为NumPy数组，然后使用extend方法将其添加到列表中。
            # all_predictions通常在循环外部初始化为空列表，用于记录整个验证集的所有预测结果。
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    print(f'Epoch {epoch + 1}/{args.epochs_num}, Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, '
          f'F1 Score: {f1:.2f}, Recall: {recall:.2f}')

# 保存模型
save_model(model, args.output_model_path)

# 测试阶段
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        know_sent_batch, mask_batch, position_batch, vm_batch, labels = process_batch(batch)
        outputs = model(know_sent_batch, mask_batch, position_batch, vm_batch)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算平均测试损失
average_test_loss = test_loss / len(test_loader)

# 计算准确率
accuracy = 100 * correct / total

# 打印验证指标
print(f'测试损失: {average_test_loss:.4f}')
print(f'准确率: {accuracy:.2f}%')