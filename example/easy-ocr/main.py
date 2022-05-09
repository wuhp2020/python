'''
 @ Libs   : python3.9 -m pip install EasyOCR -i https://mirrors.aliyun.com/pypi/simple
            python3.9 -m pip install opencv-python==4.1.2.30 -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/29
 @ Desc   : 支持40多种语言的即用型 OCR
'''

import easyocr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# need to run only once to load model into memory
reader = easyocr.Reader(lang_list=['ch_sim','en'], model_storage_directory='./model')
result = reader.readtext('1.png', detail=0)
print(result)