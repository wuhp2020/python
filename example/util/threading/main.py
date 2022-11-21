'''
Python 标准库, 更高层的线程接口
python3.8 -m pip install threading -i https://mirrors.aliyun.com/pypi/simple
'''

import threading
import time

# 线程1
def aa():
    time.sleep(0.1)
    print("开启线程: " + threading.currentThread().getName())
t1 = threading.Thread(target=aa)
t1.start()

# 线程2
class T(threading.Thread):
    def run(self):
        print("开启线程: " + threading.currentThread().getName())
t2 = T()
t2.start()

# 锁
gLock = threading.Lock()
gLock.acquire()
gLock.release()
