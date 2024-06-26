import sys

daemon=True #是否守护
bind='0.0.0.0:8000'#绑定
pidfile='gunicorn.pid'#pid文件地址
chdir='.' # 项目地址
worker_class='uvicorn.workers.UvicornWorker'
workers=1
threads=2
timeout=1200
loglevel='debug' # 日志级别
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
sys.stdout = open('logs/prints.log', 'a')
sys.stderr = open('logs/prints.log', 'a')
capture_output=True