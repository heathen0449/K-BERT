import os
import json

count = 0
true_news = 0
false_news = 0
unknown_news = 0
downloads_path = os.path.join(os.getcwd(), 'Downloads')

for file in os.listdir(downloads_path):
    if os.path.isfile(os.path.join(downloads_path, file)) and file.endswith('.jsonl'):
        print(f'正在处理{file}...')
        with open(os.path.join(downloads_path, file), 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['Result'] == 'True':
                    true_news += 1
                elif data['Result'] == 'False':
                    false_news += 1
                else:
                    unknown_news += 1
                count += 1

print(f'总共有{count}条新闻，其中{true_news}条为多主题新闻，{false_news}条为单主题新闻，{unknown_news}条未知新闻。')
print(f'多主题新闻占比：{true_news / count * 100:.2f}%')
print(f'单主题新闻占比：{false_news / count * 100:.2f}%')
print(f'二者比值大约为：{true_news / false_news:.2f}')