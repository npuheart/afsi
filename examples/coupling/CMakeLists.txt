


set(LOCAL_LIB coupling)

find_package(DOLFINX REQUIRED)
find_package(TBB REQUIRED)

# 生成 ${LOCAL_LIB} 静态库
add_library(${LOCAL_LIB} STATIC main.cpp)
target_link_libraries(${LOCAL_LIB} PUBLIC dolfinx)
target_compile_options(${LOCAL_LIB} PRIVATE -fPIC)
target_link_libraries(${LOCAL_LIB} PRIVATE TBB::tbb)



# target_compile_options(${LOCAL_LIB}  PRIVATE
#     -fsanitize=address
#     -fno-omit-frame-pointer
#     -g
#     -O1
# )

# target_link_options(${LOCAL_LIB}  PRIVATE
#     -fsanitize=address
# )
