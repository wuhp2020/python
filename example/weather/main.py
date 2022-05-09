'''
 @ Libs   : python3.9 -m pip install html5lib -i https://mirrors.aliyun.com/pypi/simple
 @ Libs   : python3.9 -m pip install pyecharts -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/5/3
 @ Desc   : 描述
'''

import requests
from bs4 import BeautifulSoup
# from pyecharts import Bar


data = []

def min_weather_page(url):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"}

    response = requests.get(url, headers=headers)
    text = response.content.decode("utf-8")
    # print(text)
    # soup = BeautifulSoup(text, "lxml")
    soup = BeautifulSoup(text, "html5lib")
    conMidtab = soup.find('div', class_='conMidtab')
    # print(conMidtab)
    tables = conMidtab.find_all('table')
    for table in tables:
        trs = table.find_all('tr')[2:]
        # print(trs)
        for index,tr in enumerate(trs):
            tds = tr.find_all('td')
            city_td = tds[0]
            if index == 0:
                city_td = tds[1]
            city = list(city_td.stripped_strings)[0]
            # print(city)
            min_weather_td = tds[-2]
            min_weather = list(min_weather_td.stripped_strings)[0]
            print({'city': city, 'min_weather': min_weather})
            data.append({'city': city, 'min_weather': int(min_weather)})
        # break


if __name__ == '__main__':
    min_weather_page("http://www.weather.com.cn/textFC/hb.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/db.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/hd.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/hz.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/hn.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/xb.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/xn.shtml")
    min_weather_page("http://www.weather.com.cn/textFC/gat.shtml")

    # data.sort(key=lambda  x: x['min_weather'])
    # print(data[0:10])