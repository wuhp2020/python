'''
 @ Libs   : python3.9 -m pip install errbot -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/1/28
 @ Desc   : 实现 ChatOps 的最简单最受欢迎的聊天机器人
'''

from errbot import BotPlugin, botcmd

class Hello(BotPlugin):
    """Example 'Hello, world!' plugin for Errbot"""

    @botcmd
    def hello(self, msg, args):
        """Return the phrase "Hello, world!" to you"""
        return "Hello, world!"