'''
python操作elasticsearch
python3.8 -m pip install elasticsearch -i https://mirrors.aliyun.com/pypi/simple
'''

from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=["http://username:password@192.168.221.131:9200/"])

# es中查询数据并回显
result = es.search(index="xxx", doc_type="xx", body={"query": {"match_all": {}}}, filter_path=['hits.hits._source'])
print(result)

# 插入数据
es.index(index = "test", doc_type = "_doc", id = 1, body = {"id":1, "name":"小明"})
# 可以不用指定id, create会自动添加id
es.create(index="test", doc_type = "_doc",id = 2, body = {"id":2, "name":"小红"})

# 删除指定数据
es.delete(index='test', doc_type='_doc', id=1)

# 修改字段
es.update(index = "test", doc_type = "_doc", id = 1, body = {"doc":{"name":"张三"}})

# 查询数据
es.get(index = "test", doc_type = "_doc", id = 1)
es.search(index = "test", doc_type = "_doc", body = {})

# 滚动分页的func
es.scroll(index = "test", scroll_id = "scroll_id", scroll = "5m")
