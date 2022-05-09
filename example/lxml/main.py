'''
 @ Libs   : python3.9 -m pip install lxml -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/2/4
 @ Desc   : 一个非常快速，简单易用, 功能齐全的库, 用来处理 HTML 和 XML
'''

from lxml import etree
import requests
import os
from urllib import request

def page_imgs(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
    }

    response = requests.get(url=url, headers=headers)
    text = response.content
    # print(text.decode('utf-8'))
    html = etree.HTML(text.decode('utf-8'))
    # 获取属性
    imgs = html.xpath('//img[@class="ui image lazy"]')
    for img in imgs:
        try:
            # 获取值
            img_url = img.get('data-original')
            title = img.get('title')
            suffix = os.path.splitext(img_url)[1]
            filename = (title + suffix).replace("(", "").replace(")", "").replace("#", "").replace("*", "").replace("!", "").replace("（", "").replace("）", "").replace("！", "").replace("/", "").replace(",", "").replace("，", "")
            print(img_url)
            request.urlretrieve(img_url, './images/'+ filename)
        except:
            print("error, skip ================= ")

if __name__ == '__main__':
    for pageNo in range(200):
        page_imgs('https://fabiaoqing.com/biaoqing/lists/page/' + str(pageNo) + '.html')