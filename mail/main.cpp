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