'''
轻量级Web应用框架, 相比Django和Pyramid, 它也被称为微框架.
使用Flask开发Web应用十分方便, 甚至几行代码即可建立一个小型网站.
Flask核心十分简单, 并不直接包含诸如数据库访问等的抽象访问层, 而是通过扩展模块形式来支持
python3.8 -m pip install flask -i https://mirrors.aliyun.com/pypi/simple
'''

import flask

app = flask.Flask(__name__)

@app.route('/')
def index():
    return "这是一个api"


app.run(host='0.0.0.0', port=8008, threaded=True)
