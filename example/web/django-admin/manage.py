'''
一个开放源代码的Web应用框架, 由Python写成. 是Python生态中最流行的开源Web应用框架, Django采用模型、模板和视图的编写模式, 称为MTV模式
python3.8 -m pip install django -i https://mirrors.aliyun.com/pypi/simple
python3 manage.py runserver
'''

import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django-admin.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
