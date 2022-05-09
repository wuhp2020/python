'''
 @ Libs   : python3.9 -m pip install jieba -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/4/10
 @ Desc   : 描述
'''

import os

base_path = '/Volumes/wuhp/pan.baidu/01.尚硅谷-Go语言核心编程/'
path_list = os.listdir('/Volumes/wuhp/pan.baidu/01.尚硅谷-Go语言核心编程')
for file in path_list:
    if not file.startswith('.') \
            and not file.startswith('资料') \
            and not file.startswith('代码') \
            and not file.startswith('软件')\
            and not file.startswith('笔记')\
            and not file.startswith('视频'):
        old_temp = os.path.basename(file)
        old = base_path + os.path.basename(file)
        new = base_path + old_temp[0:3] + old_temp[-4:]
        # print(new)
        os.rename(old, new)