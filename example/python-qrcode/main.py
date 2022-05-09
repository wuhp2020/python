'''
 @ Libs   : python3.9 -m pip install qrcode -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/28
 @ Desc   : 一个纯 Python 实现的二维码生成器
'''

import qrcode

img = qrcode.make('wuhp')
img.save('./wuhp.png')