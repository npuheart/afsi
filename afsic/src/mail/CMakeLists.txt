cmake_minimum_required(VERSION 3.28)

project("test_email")

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/bin)

find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

# 生成 smtp_mail_sender 静态库
add_library(smtp_mail_sender STATIC smtp_mail_sender.cpp)
target_link_libraries(smtp_mail_sender PRIVATE OpenSSL::SSL OpenSSL::Crypto CURL::libcurl)

# 可执行文件只包含 main.cpp，并链接 smtp_mail_sender 库
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE smtp_mail_sender)