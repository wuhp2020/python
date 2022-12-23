'''
支持40多种语言的即用型 OCR
python3.5 -m pip install opencv-python-headless==3.4.18.65 -i https://mirrors.aliyun.com/pypi/simple
python3.5 -m pip install opencv-python==3.4.5.20 -i https://mirrors.aliyun.com/pypi/simple
python3.5 -m pip install EasyOCR==1.3.2 -i https://mirrors.aliyun.com/pypi/simple
'''

import ssl
import sys
import os
import easyocr

ssl._create_default_https_context = ssl._create_unverified_context
ocrreader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory='./model')
# 默认阈值
threshold = 0.1

for i in range(1, len(sys.argv)):  # 获取命令行参数：argv[0]表示可执行文件本身
    imgfile = sys.argv[i]  # 待识别文件名
    imgfilext = os.path.splitext(imgfile)[-1]  # 文件后缀名
    if imgfilext.upper() not in ['.JPG', '.JPEG', '.PNG', '.BMP']:  # 转换为大写后再比对
        print('\t', imgfile, ' 不是有效图片格式(jpg/jpeg/png/bmp)!')
        continue
    result = ocrreader.readtext(imgfile)
    paper = ''
    for w in result:
        if w[2] > threshold:  # 设置一定的置信度阈值
            paper = paper + w[1]
    paper.replace(' ', '').replace('_', '').replace('^', '').replace('~', '').replace('`', '').replace('&',
                                                                                                       '')  # 删除无效数据

    # 记录当前文件的识别结果，保存为同名的txt文件
    newfname = os.path.splitext(imgfile)[0] + '.txt'  # 与原文件同名的txt文件（包括目录）
    # newfname=os.path.splitext(imgfile)[0].split('/')[-1].split('\\')[-1]+'.txt'#与原文件同名的txt文件（不包括目录，仅文件名）
    try:
        with open(newfname, 'w') as txtfile:
            txtfile.write(paper)
    except(Exception) as e:
        print('\t', newfname, ' OCR Error: ', e)  # 输出异常错误
        continue