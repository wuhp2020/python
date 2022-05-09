'''
 @ Libs   : python3.9 -m pip install requests -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/29
 @ Desc   : 人性化的 HTTP 请求库
'''

import requests

url = 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip'
proxy = {
    'http://': '47.92.234.75:80'
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'
}
file_name = '../easy-ocr/model/zh_sim_g2.zip'

resp = requests.get(url=url, headers=headers, proxies=proxy, stream=True)
with open(file_name, "wb") as file:
    chunk_size = 1024
    current_size = 0
    total = int(resp.headers['content-length'])  # 内容体总大小
    for data in resp.iter_content(chunk_size=chunk_size):
        file.write(data)
        print('======================')
        current_size += len(data)
        print('total: ' + str(total), 'current: ' + str(current_size), 'progress: ' + str(current_size/total))

