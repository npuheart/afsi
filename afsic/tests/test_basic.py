import afsic as m

def test_add():
    assert m.add(1, 2) == 3

import os
from afsic import EmailInfo, send_email

info = EmailInfo()
info.smtp_url = "smtp://smtp.qq.com:587"
info.username = "499908174@qq.com"
info.password = os.getenv("SMTP_QQ") 
info.From = "499908174@qq.com"
info.to = "mapengfei@mail.nwpu.edu.cn"
info.subject = "Test"
info.body = "Hello from Python"
info.is_html = False
send_email(info)


# 流体求解器
from afsic import IPCSSolver