'''
python3.8 -m pip install numpy -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install fastapi -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install uvicorn -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install opencv-python==3.4.18.65 -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install dlib==19.13.1 -i https://mirrors.aliyun.com/pypi/simple
'''

import dlib
import cv2
import base64
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI(title="人脸检测",
              description="人脸检测",
              docs_url="/docs",
              openapi_url="/openapi")

facerec = dlib.face_recognition_model_v1(
    "./model/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

class Params(BaseModel):
    imageBase64: str

@app.post("/api/v1/face/detect")
async def detect(params: Params):

    # 检测是否有数据
    if not params.imageBase64:
        return ('no params, fail')

    img = base64.b64decode(params.imageBase64)
    # 转换np序列
    img_array = np.fromstring(img, np.uint8)
    # 转换灰度图
    img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    faces = detector(img_gray, 0)
    if len(faces) == 1:
        return {"code": "200", "data": [faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()]}
    else:
        return {"code": "500", "msg": "no face or more"}

if __name__ == '__main__':

    uvicorn.run("main:app", host="0.0.0.0", port=8009)