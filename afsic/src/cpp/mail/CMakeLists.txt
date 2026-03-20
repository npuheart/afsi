find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

add_library(smtp_mail_sender STATIC smtp_mail_sender.cpp)
target_link_libraries(smtp_mail_sender PRIVATE OpenSSL::SSL OpenSSL::Crypto CURL::libcurl)
target_compile_options(smtp_mail_sender PRIVATE -fPIC)