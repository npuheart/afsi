cmake_minimum_required (VERSION 3.12)

project ("test_email")

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/bin)

# 看情况修改
set(CMAKE_PREFIX_PATH "D:/vcpkg-master/installed/x64-windows")

find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

add_executable (${PROJECT_NAME} "main.cpp" "smtp_mail_sender.cpp")

target_link_libraries(${PROJECT_NAME} PRIVATE 
    OpenSSL::SSL 
    OpenSSL::Crypto 
    CURL::libcurl # linux 直接链 curl
)