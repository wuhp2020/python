'''
支持40多种语言的即用型 OCR
python3.8 -m pip install EasyOCR -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install opencv-python==4.1.2.30 -i https://mirrors.aliyun.com/pypi/simple
'''

import easyocr
import ssl
import os
import re

ssl._create_default_https_context = ssl._create_unverified_context

# need to run only once to load model into memory
reader = easyocr.Reader(lang_list=['ch_sim','en'], model_storage_directory='./model')

reg = '(13\d{9}|14[5|7]\d{8}|15\d{9}|166{\d{8}|17[3|6|7]{\d{8}|18\d{9})'

path = '/Users/mac/Desktop/data/'
with open("/Users/mac/Desktop/data.txt", "w") as f:
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            print('ok')
        else:
            break
        print(path + file)
        result = reader.readtext(path + file, detail=0)

        for content in result:
            f.write(content.replace('\r','').replace('\n','') + '\t')
            print(content.replace('\r','').replace('\n',''), end='')
            print('\t', end='')
            if len(re.findall(reg, content)) > 0 and len(content) == 11:
                print('\n')
                f.write('\n')
