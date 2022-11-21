'''
中文分词模块
python3.8 -m pip install jieba -i https://mirrors.aliyun.com/pypi/simple
'''

import jieba

counts = {}
words = jieba.lcut('中国人, 是好人, 我们都是中国人')
for word in words:
    counts[word] = counts.get(word, 0) + 1


sorted(counts.items(), key=lambda x: x[1], reverse=True)
for key in counts:
    print(key, counts[key])