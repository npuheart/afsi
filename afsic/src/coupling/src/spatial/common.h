/// @date 2024-09-24
/// @file common.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief 共用的 Index 类型，用于表示多维索引
/// 
///
#pragma once
#include <vector>
#include <type_traits>

// 主模板声明
template <size_t dim, typename Enable = void>
class Index;

// 特化 dim = 1
template <size_t dim>
class Index<dim, typename std::enable_if<dim == 1>::type>
{
public:
    Index(size_t i) : i(i) {}
    size_t i;
};

// 特化 dim = 2
template <size_t dim>
class Index<dim, typename std::enable_if<dim == 2>::type>
{
public:
    Index(size_t i, size_t j) : i(i), j(j) {}
    size_t i, j;
};

// 特化 dim = 3
template <size_t dim>
class Index<dim, typename std::enable_if<dim == 3>::type>
{
public:
    Index(size_t i, size_t j, size_t k) : i(i), j(j), k(k) {}
    size_t i, j, k;
};

// Index{1,2,3}
// // 默认模板，处理不匹配的情况
// template <size_t dim, typename Enable>
// class Index
// {
// public:
//     Index() = delete;  // 禁用默认构造函数
// };

// int main() {
//     Index<1> idx1(5);
//     std::cout << "Index<1>: " << idx1.i << std::endl;

//     Index<2> idx2(3, 4);
//     std::cout << "Index<2>: " << idx2.i << ", " << idx2.j << std::endl;

//     Index<3> idx3(1, 2, 3);
//     std::cout << "Index<3>: " << idx3.i << ", " << idx3.j << ", " << idx3.k << std::endl;

//     // Index<4> idx4;  // 编译错误，因为没有定义 Index<4> 特化

//     return 0;
// }
