'''
python3.8 -m pip install fastapi -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install uvicorn -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install opencv-python-headless==3.4.18.65 -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install opencv-python==3.4.18.65 -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install EasyOCR==1.3.2 -i https://mirrors.aliyun.com/pypi/simple
'''

import ssl
import easyocr
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import base64

app = FastAPI(title="文字提取",
              description="文字提取",
              docs_url="/docs",
              openapi_url="/openapi")

ssl._create_default_https_context = ssl._create_unverified_context
ocrreader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory='./model')
# 默认阈值
threshold = 0.1

class Params(BaseModel):
    imageBase64: str

@app.post("/api/v1/ocr")
async def ocr(params: Params):

    # 检测是否有数据
    if not params.imageBase64:
        return ('no params, fail')

    img = base64.b64decode(params.imageBase64)
    result = ocrreader.readtext(img)
    paper = ''
    for w in result:
        # 设置一定的置信度阈值
        if w[2] > threshold:
            paper = paper + w[1]
    paper.replace(' ', '').replace('_', '').replace('^', '').replace('~', '').replace('`', '').replace('&', '')
    return {"code": "200", "data": paper}

if __name__ == '__main__':

    uvicorn.run("main:app", host="0.0.0.0", port=8008)