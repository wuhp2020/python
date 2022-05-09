# encoding=utf-8
'''
 @ Libs   : python2.7 -m pip install tgrocery -i https://mirrors.aliyun.com/pypi/simple
 @ Libs   : python2.7 -m pip install converter -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/28
 @ Desc   : 一简单高效的短文本分类工具, 基于 LibLinear 和 Jieba
'''

from tgrocery import Grocery

grocery = Grocery('sample')

# 训练
train_src = [
('education', '名师指导托福语法技巧：名词的复数形式'),
('education', '中国高考成绩海外认可 是“狼来了”吗？'),
('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与')
]

grocery.train(train_src)
grocery.save()

# 加载模型
new_grocery = Grocery('sample')
new_grocery.load()

# 预测
print(new_grocery.predict('考生必读：新托福写作考试评分标准'))
