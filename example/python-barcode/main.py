'''
 @ Libs   : python3.9 -m pip install python-barcode -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/28
 @ Desc   : 不借助其他库在 Python 程序中生成条形码
'''

import barcode

bar = barcode.get('ean13', '123456789011')
filename = bar.save('ean13')
print(filename)
