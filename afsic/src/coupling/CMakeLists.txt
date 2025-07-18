


set(LOCAL_LIB coupling)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

# 生成 ${LOCAL_LIB} 静态库
add_library(${LOCAL_LIB} STATIC main.cpp)
target_link_libraries(${LOCAL_LIB} PRIVATE OpenSSL::SSL OpenSSL::Crypto CURL::libcurl)
target_compile_options(${LOCAL_LIB} PRIVATE -fPIC)
