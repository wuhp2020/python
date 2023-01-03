'''
python3.8 -m pip install pypinyin -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install fastapi -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install uvicorn -i https://mirrors.aliyun.com/pypi/simple
'''

import pypinyin
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="姓名识别",
              description="姓名识别",
              docs_url="/docs",
              openapi_url="/openapi")

class Params(BaseModel):
    name: str

@app.post("/api/v1/pinyin/name")
async def detect(params: Params):

    # 检测是否有数据
    if not params.name:
        return ('no params, fail')

    listPinYin = pypinyin.pinyin(params.name, heteronym=True, style=pypinyin.NORMAL)
    result = []
    assemble(listPinYin, result=result)
    return {"code": "200", "data": result}

def assemble(lis, jude=True, result=[]):
    if jude: lis = [[[i] for i in lis[0]]] + lis[1:]
    if len(lis) > 2:
        for i in lis[0]:
            for j in lis[1]:
                assemble([[i + [j]]] + lis[2:], False, result=result)
    elif len(lis) == 2:
        for i in lis[0]:
            for j in lis[1]:
                result.append(i + [j])

if __name__ == '__main__':

    uvicorn.run("main:app", host="0.0.0.0", port=8009)


