#include "smtp_mail_sender.h"
#include <curl/curl.h>
#include <iostream>
#include <cstring>

namespace smtp {

    struct UploadStatus {
        std::size_t bytes_read = 0;
        std::string payload;
    };

    static std::size_t payload_source(char* ptr, std::size_t size, std::size_t nmemb, void* userp) {
        UploadStatus* upload_ctx = static_cast<UploadStatus*>(userp);
        const std::size_t buffer_size = size * nmemb;

        if (upload_ctx->bytes_read >= upload_ctx->payload.size())
            return 0;

        std::size_t copy_size = upload_ctx->payload.size() - upload_ctx->bytes_read;
        if (copy_size > buffer_size)
            copy_size = buffer_size;

        memcpy(ptr, upload_ctx->payload.c_str() + upload_ctx->bytes_read, copy_size);
        upload_ctx->bytes_read += copy_size;
        return copy_size;
    }

    bool send_email(const EmailInfo& info) {
        CURL* curl = curl_easy_init();
        if (!curl) return false;

        CURLcode res;
        curl_slist* recipients = nullptr;

        // 构造邮件内容
        std::string payload;
        payload += "To: " + info.to + "\r\n";
        payload += "From: " + info.from + "\r\n";
        payload += "Subject: " + info.subject + "\r\n";
        if (info.is_html)
            payload += "Content-Type: text/html; charset=UTF-8\r\n";
        else
            payload += "Content-Type: text/plain; charset=UTF-8\r\n";
        payload += "\r\n";
        payload += info.body + "\r\n";

        UploadStatus upload_ctx = { 0, payload };

        curl_easy_setopt(curl, CURLOPT_URL, info.smtp_url.c_str());                 // 设置SMTP服务器地址
        curl_easy_setopt(curl, CURLOPT_PORT, 587L);                                 // 设置SMTP端口
        curl_easy_setopt(curl, CURLOPT_USERNAME, info.username.c_str());            // 设置SMTP用户名
        curl_easy_setopt(curl, CURLOPT_PASSWORD, info.password.c_str());            // 设置SMTP密码（授权码）
        curl_easy_setopt(curl, CURLOPT_MAIL_FROM, ("<" + info.from + ">").c_str()); // 设置发件人邮箱
        recipients = curl_slist_append(nullptr, ("<" + info.to + ">").c_str());     // 构造收件人邮箱列表
        curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);                      // 设置收件人邮箱
        curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);                    // 启用SSL/TLS加密传输               
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);                         // 跳过SSL证书校验
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);                         // 跳过SSL主机名校验
        curl_easy_setopt(curl, CURLOPT_LOGIN_OPTIONS, "AUTH=LOGIN");                // 指定 LOGIN 认证方式
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, payload_source);               // 设置回调函数，用于提供邮件内容读取
        curl_easy_setopt(curl, CURLOPT_READDATA, &upload_ctx);                      // 设置回调函数的上下文数据，传递upload_ctx
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);                                 // 设置为上传模式（即发邮件）
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);                                // 开启调试信息输出

        res = curl_easy_perform(curl);                                              // 正式执行SMTP流程（连接、认证、发信）。libcurl会自动完成所有SMTP细节
        curl_slist_free_all(recipients);                                            // 释放收件人列表内存
        curl_easy_cleanup(curl);                                                    // 清理CURL句柄

        if (res != CURLE_OK) {
            std::cerr << "❌ curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            return false;
        }

        return true;
    }

} // namespace smtp
