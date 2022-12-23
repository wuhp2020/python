'''
python3.8 -m pip install fastapi -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install uvicorn -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install python-multipart -i https://mirrors.aliyun.com/pypi/simple
'''

from fastapi import FastAPI, File, UploadFile
import uvicorn
from typing import Optional, List, Dict

app = FastAPI(title="fastapi文档",
              description="fastapi文档",
              docs_url="/docs",
              openapi_url="/openapi")

@app.get("/api/v1/{id}")
async def get_item(id: int):
    # Flask定义类型是在路由当中, 也就是在<>里面, 变量和类型通过:分隔
    # 而FastAPI是使用类型注解的方式, 此时的id要求一个整型(准确的说是一个能够转成整型的字符串)
    return {"id": id}

@app.post("/api/v1/detect")
async def detect(dict: Dict):
    # dict就是我们接收的请求体, 它需要通过json来传递
    # 通过dict.xxx的方式我们可以获取和修改内部的所有属性
    # 直接返回对象也是可以的
    return dict

@app.post("/api/v1/file")
async def file1(file: UploadFile = File(...)):
    return f"文件名: {file.filename}, 文件大小: {len(await file.read())}"

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8008)
