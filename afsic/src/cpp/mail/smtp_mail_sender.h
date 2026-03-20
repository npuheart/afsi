#pragma once
#include <string>

namespace smtp {

    struct EmailInfo {
        std::string smtp_url;   // SMTP地址，例如 smtp://smtp.qq.com:587
        std::string username;   // 邮箱账号
        std::string password;   // 邮箱密码（授权码）
        std::string from;       // 发件人邮箱
        std::string to;         // 收件人邮箱
        std::string subject;    // 邮件主题
        std::string body;       // 邮件内容（纯文本或HTML）
        bool is_html = false;   // 是否HTML格式
    };

    // 发送邮件，支持纯文本与HTML
    bool send_email(const EmailInfo& info);

} // namespace smtp