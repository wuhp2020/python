'''
python操作elasticsearch
python3.8 -m pip install elasticsearch==7.7.0 -i https://mirrors.aliyun.com/pypi/simple
'''

from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=["http://192.168.221.131:9200"])


# 查询
res = es.search(index="test", body={"query":{"match_all":{}}})
for hit in res['hits']['hits']:
    print(hit)

# 插入数据
# 可以不用指定id, index会自动添加id
es.index(index="test", doc_type="doc", body={"name":"小红"})

# 通过id修改字段
es.update(index="test", doc_type="doc", id='TNM_mYQB9c0QNmdEbgBh', body={"doc":{"name":"张三"}})

# 通过id查询数据
doc = es.get(index="test", doc_type="doc", id='TNM_mYQB9c0QNmdEbgBh')
print(doc)

# es中查询数据并回显
result = es.search(index="test", doc_type="doc", body={"query": {"match_all": {}}}, filter_path=['hits.hits._source'])
print(result)

# 删除指定数据
for hit in res['hits']['hits']:
    es.delete(index='test', doc_type='doc', id=hit['_id'])
