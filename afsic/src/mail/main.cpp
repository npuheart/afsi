#include "smtp_mail_sender.h"

int main() {
    smtp::EmailInfo info;

    const char* pwd = std::getenv("SMTP_QQ");
    if (pwd) {
        info.password = pwd;
    } else {
        info.password = "";
    }

    info.smtp_url = "smtp://smtp.qq.com:587";
    info.username = "499908174@qq.com";
    info.from = "499908174@qq.com";
    info.to = "mapengfei@mail.nwpu.edu.cn";
    info.subject = "HTML 邮件";
    info.body = R"(
        <html><body><h1>你好！</h1><p>这是一封<b>HTML</b>邮件。</p></body></html>
    )";
    info.is_html = true;

    smtp::send_email(info);

    info.subject = "纯文本邮件";
    info.body = "这是一封纯文本邮件。";
    smtp::send_email(info);
}
/*
要在 Python 中调用 C++ 的 smtp::send_email，可以通过 nanobind 包装 smtp::send_email 函数和 smtp::EmailInfo 结构体。大致步骤如下：

1. 用 nanobind 包装 smtp::send_email 和 smtp::EmailInfo。
2. 编译为 Python 可导入的模块（如 smtp_mail_sender.so）。
3. 在 Python 中 import 并调用。

示例 nanobind 包装代码（假设 smtp_mail_sender.h/.cpp 已经实现 send_email 和 EmailInfo）：

#include <nanobind/nanobind.h>

编译后，在 Python 中这样用：

import smtp_mail_sender

info = smtp_mail_sender.EmailInfo()
info.smtp_url = "smtp://smtp.qq.com:587"
info.username = "499908174@qq.com"
info.password = "your_password"
info.from = "499908174@qq.com"
info.to = "mapengfei@mail.nwpu.edu.cn"
info.subject = "Test"
info.body = "Hello from Python"
info.is_html = False

smtp_mail_sender.send_email(info)

详细 nanobind 教程可参考官方文档：https://nanobind.readthedocs.io/
*/
