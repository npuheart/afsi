/// @date 2024-09-24
/// @file kernel_helper.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief Kernel 的辅助程序 1. 计算 Kernel 的基节点 2. Kernel 升维
/// 
///
#pragma once
#include <iostream>
#include <cmath>
#include <array>
#include "kernel_expression.h"

namespace MATH_TOOLS
{
    /**
       int floor much faster than (int)floor(x)
     */
    inline static int int_floor(float x)
    {
        int i = (int)x;     /* truncate */
        return i - (i > x); /* convert trunc to floor */
    }

    inline static int int_floor(double x)
    {
        int i = (int)x;     /* truncate */
        return i - (i > x); /* convert trunc to floor */
    }

}

template <class T, size_t dim, template <typename, size_t> class Vector>
Vector<T, dim> multiply(const Vector<T, dim> &b, const Vector<T, dim> &c) 
{
    Vector<T, dim> a;
    for (size_t i = 0; i < dim; i++)
        a[i] = b[i] * c[i];
    return a;
}

template <class T, size_t dim, template <typename, size_t> class Vector>
Vector<T, dim> multiply(const Vector<T, dim> &b, T c) 
{
    Vector<T, dim> a;
    for (size_t i = 0; i < dim; i++)
        a[i] = b[i] * c;
    return a;
}

template <class T, size_t dim, template <typename, size_t> class Vector>
Vector<T, dim> multiply(T c, const Vector<T, dim> &b)
{
    return multiply(b, c);
}

template <class T, size_t dim, template <typename, size_t> class Vector>
Vector<T, dim> one_over(const Vector<T, dim> &a)
{
    Vector<T, dim> b;
    for (size_t i = 0; i < dim; i++)
        b[i] = 1.0/a[i];
    return b;
}

template <size_t X, size_t Y=0, size_t Z=0> constexpr
size_t octal_to_decimal(){
    static_assert(X > 0 && X < 8 && Y < 8 && Z < 8, "X, Y, Z must < 8");
    return Z * 64 + Y * 8 + X;
}

template <size_t hash, size_t position> constexpr
size_t decimal_to_octal(){
    static_assert(hash < 512, "hash must < 512");
    static_assert(position == 0 || position == 1 || position == 2, "position must 0, 1 or 2");
    if constexpr (position == 0){
        return hash % 8;
    }
    if constexpr (position == 1){
        return hash / 8 % 8;
    }
    if constexpr (position == 2){
        return hash / 64 % 8;
    }
}

template <size_t hash>
struct PlaceValue {
    static constexpr size_t _0 = decimal_to_octal<hash, 0>();
    static constexpr size_t _1 = decimal_to_octal<hash, 1>();
    static constexpr size_t _2 = decimal_to_octal<hash, 2>();
    static constexpr size_t _sum = _0 + _1 + _2;
};

template <size_t interpolate_width, class T>
inline int baseNode(const T &x)
{
    static_assert(interpolate_width >= 2, "interpolate_width must >= 2");
    static_assert(interpolate_width <= 7, "interpolate_width must <= 7");
    return MATH_TOOLS::int_floor(x - (T)0.5 * (interpolate_width - 2));
}

template <class PlaceValue, class T, size_t dim, template <typename, size_t> class Vector>
inline Vector<int, dim> baseNode(const Vector<T, dim> &x)
{
    Vector<int, dim> base;

    if constexpr (dim == 1) {
        base[0] = baseNode<PlaceValue::_0>(x[0]);
    } else if constexpr (dim == 2) {
        base[0] = baseNode<PlaceValue::_0>(x[0]);
        base[1] = baseNode<PlaceValue::_1>(x[1]);
    } else if constexpr (dim == 3) {
        base[0] = baseNode<PlaceValue::_0>(x[0]);
        base[1] = baseNode<PlaceValue::_1>(x[1]);
        base[2] = baseNode<PlaceValue::_2>(x[2]);
    }
    return base;
}

template <class PlaceValue, class T, size_t dim, template <typename, size_t> class Vector> 
struct IBKernel
{
    using kernel_width = PlaceValue;

    const Vector<T, dim> dh;
    const Vector<T, dim> one_over_dh;
    Vector<int, dim> base_node;
    Vector<T, PlaceValue::_sum> w;
    Vector<T, PlaceValue::_sum> dw;
    Vector<T, PlaceValue::_sum> ddw;

    IBKernel(const Vector<T, dim> &X, const Vector<T, dim>& dh)
        : dh{dh}
        , one_over_dh(one_over(dh))
    {
        compute(X);
    }

    void compute(const Vector<T, dim> &X)
    {
        // printf("dh        : %f %f %f\n", dh[0], dh[1], dh[2]);
        Vector<T, dim> X0 = multiply(one_over_dh, X);
        base_node = baseNode<PlaceValue>(X0);
        // printf("base_node : %6ld %6ld\n", base_node[0], base_node[1]);
        // printf("X0        : %f %f %f\n", X0[0], X0[1], X0[2]);

        if constexpr (dim == 1){
            evaluate_kernel_peskin<T, PlaceValue::_0>(X0[0]-base_node[0], &w[0]);
        }
        if constexpr (dim == 2)
        { 
            evaluate_kernel_peskin<T, PlaceValue::_0>(X0[0]-base_node[0], &w[0],             &dw[0]);
            evaluate_kernel_peskin<T, PlaceValue::_1>(X0[1]-base_node[1], &w[PlaceValue::_0],&dw[PlaceValue::_0]);
        }
        if constexpr (dim == 3)
        {
            // printf("base_node : %6d %6d %6d\n", base_node[0], base_node[1], base_node[2]);
            // printf("X0        : %6.1f %6.1f %6.1f\n", X0[0], X0[1], X0[2]);
            evaluate_kernel_peskin<T, PlaceValue::_0>(X0[0]-base_node[0], &w[0]);
            evaluate_kernel_peskin<T, PlaceValue::_1>(X0[1]-base_node[1], &w[PlaceValue::_0]);
            evaluate_kernel_peskin<T, PlaceValue::_2>(X0[2]-base_node[2], &w[PlaceValue::_0 + PlaceValue::_1]);
        }
    }
};



// int main(){


//     {
//         std::array<double, 1> dh = {0.24};
//         auto one_over_dh = one_over(dh);
//         printf("one_over_dh: %f\n", one_over_dh[0]);
//     }
//     {
//         std::array<double, 2> dh = {0.5, 0.4};
//         auto one_over_dh = one_over(dh);
//         printf("one_over_dh: %f, %f\n", one_over_dh[0], one_over_dh[1]);
//     }
//     {
//         std::array<double, 3> dh = {0.5, 0.6,0.3};
//         std::array<double, 3> x = {1.0, 2.0, 3.0};
//         auto x0 = multiply(one_over(dh), x);
//         auto m1 = multiply(dh, 2.0);
//         auto m2 = multiply(2.0, dh);
//         printf("x0: %f, %f, %f\n", x0[0], x0[1], x0[2]);
//         printf("m1: %f, %f, %f\n", m1[0], m1[1], m1[2]);
//         printf("m2: %f, %f, %f\n", m2[0], m2[1], m2[2]);
//     }

//     {
//         const size_t size = octal_to_decimal<3,4,5>();
//         printf("size: %ld\n", size);
//         printf("x: %ld, y: %ld, z: %ld\n", decimal_to_octal<size, 0>(),decimal_to_octal<size, 1>(),decimal_to_octal<size, 2>());
//     }
//     {
//         PlaceValue<octal_to_decimal<3,4,5>()> pv;
//         printf("pv._0: %ld\n", pv._0);
//         printf("pv._0: %ld\n", PlaceValue<octal_to_decimal<3,4,5>()>::_0);
//         printf("pv._1: %ld\n", PlaceValue<octal_to_decimal<3,4,5>()>::_1);
//         printf("pv._2: %ld\n", PlaceValue<octal_to_decimal<3,4,5>()>::_2);
//         printf("pv._sum: %ld\n", PlaceValue<octal_to_decimal<3,4,5>()>::_sum);
//     }
//     {
//         constexpr size_t dim = 2;
//         constexpr size_t kernel_width = 2;

//         constexpr size_t kernel_width_x = 2;
//         constexpr size_t kernel_width_y = 3;
//         constexpr size_t kernel_width_z = 4;

//         using PV = PlaceValue<octal_to_decimal<kernel_width_x,kernel_width_y,kernel_width_z>()>;
//         using IBW = IBKernel<PV, double, dim, std::array>;   

//         IBW ibw({5.3, 5.6}, {0.5, 0.6});
        
//         printf("ibw.base_node[0,1] = %ld %ld\n", ibw.base_node[0], ibw.base_node[1]);
//     }
    
    
//     return 0;
// }

