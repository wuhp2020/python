'''
简单易用的 python 人脸识别库
python3.5 -m pip install numpy -i https://mirrors.aliyun.com/pypi/simple
python3.5 -m pip install flask -i https://mirrors.aliyun.com/pypi/simple
python3.5 -m pip install opencv-python==3.4.5.20 -i https://mirrors.aliyun.com/pypi/simple
python3.5 -m pip install dlib==1.0.3 -i https://mirrors.aliyun.com/pypi/simple
'''

import dlib
import cv2
import base64
from flask import Flask, request, jsonify
import json
import numpy as np

app = Flask(__name__)

facerec = dlib.face_recognition_model_v1(
    "./model/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

@app.route('/api/v1/face/detect', methods=['POST'])
def detect():

    if not request.data:   #检测是否有数据
        return ('no params, fail')
    params = request.data.decode('utf-8')
    # 获取到POST过来的数据
    params_obj = json.loads(params)

    img = base64.b64decode(params_obj['imageBase64'])
    img_array = np.fromstring(img, np.uint8)  # 转换np序列
    img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 转换灰度图

    faces = detector(img_gray, 0)
    if len(faces) == 1:
        return jsonify({"code": "200", "data": [faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()]})
    else:
        return jsonify({"code": "500", "msg": "no face or more"})


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8990, threaded=True)