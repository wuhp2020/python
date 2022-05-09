'''
 @ Libs   : python3.9 -m pip install py2app -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/28
 @ Desc   : 将 Python 脚本变为独立软件包Mac OS X
'''

from tkinter import *


def on_click():
    label['text'] = text.get()


root = Tk(className='bitunion')
label = Label(root)
label['text'] = 'be on your own'
label.pack()
text = StringVar()
text.set('change to what?')
entry = Entry(root)
entry['textvariable'] = text
entry.pack()
button = Button(root)
button['text'] = 'change it'
button['command'] = on_click
button.pack()
root.mainloop()