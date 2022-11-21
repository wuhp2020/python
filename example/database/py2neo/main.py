'''
图数据库
python3.8 -m pip install py2neo==3.5.32 -i https://mirrors.aliyun.com/pypi/simple
'''

from py2neo import Graph, Node, Relationship,NodeMatcher
graph = Graph('http://192.168.221.131:7474', auth=('neo4j', '123'))

# 头实体
head = Node("regoin", name='邯郸市')
# 尾实体
tail = Node("regoin", name='河北省')
# 头尾实体关系
entity = Relationship(head,"属于", tail)
# 创建实例
graph.create(entity)

# 头实体
head = Node("regoin", name='丛台区')
# 尾实体
tail = Node("regoin", name='河北省')
# 头尾实体关系
entity = Relationship(head,"属于", tail)
# 创建实例
graph.create(entity)
