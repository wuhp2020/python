'''
操作MySQL数据库
python3.8 -m pip install pymysql -i https://mirrors.aliyun.com/pypi/simple
'''

import pymysql

connect = pymysql.connect(
        host='192.168.221.131',
        port=3306,
        user='root',
        password='123',
        database='python'
)

cursor = connect.cursor()
begin = connect.begin()
connect.commit()
