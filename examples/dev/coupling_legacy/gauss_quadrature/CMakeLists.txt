

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})



# 创建静态库目标
add_library(gauss_quadrature STATIC FacetIntegration.cpp)

# Target libraries
target_link_libraries(gauss_quadrature PRIVATE dolfin)

set_source_files_properties(FacetIntegration.cpp PROPERTIES COMPILE_FLAGS "-std=c++17 -fPIC")
