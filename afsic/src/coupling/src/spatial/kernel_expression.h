/// @date 2024-09-24
/// @file kernel_expression.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief kernel 表达式, 可分三类：1. BSpline 2. Peskin 3. others
/// 
///
#pragma once
#include <iostream>

template <typename T, size_t kernel_width>
void evaluate_kernel_peskin(const T dx, T *w, T *dw = nullptr, T *ddw = nullptr)
{
    if constexpr (kernel_width == 2)
    {
        w[0] = 1 - dx;
        w[1] = dx;
        if(dw){
            dw[0] = -1.0;
            dw[1] = 1.0;
        }
    }
    if constexpr (kernel_width == 3)
    {

        // T dx = x - base_node; // r >= 0.5 && r < 1.5
        T d1 = dx - 1.0;         // r < 0.5
        T d2 = 2.0 - dx;         // r >= 0.5 && r < 1.5

        w[0] = 1.0 / 6.0 * (5.0 - 3.0 * dx - std::sqrt(1.0 - 3.0 * (1.0 - dx) * (1.0 - dx)));
        w[1] = 1.0 / 3.0 * (1.0 + std::sqrt(1.0 - 3.0 * d1 * d1));        // d1
        w[2] = 1.0 / 6.0 * (5.0 - 3.0 * d2 - std::sqrt(1.0 - 3.0 * (1.0 - d2) * (1.0 - d2)));
    
    }
    if constexpr (kernel_width == 4)
    {
        // T dx = x - base_node; // r >= 1 && r < 2
        T d1 = dx - 1.0;      // r < 1
        T d2 = 2.0 - dx;      // r < 1
        T d3 = 3.0 - dx;      // r >= 1 && r < 2

        w[0] = (5.0 - 2.0 * dx - std::sqrt(-7.0 + 12.0 * dx - 4.0 * dx * dx)) / 8.0;
        w[1] = (3.0 - 2.0 * d1 + std::sqrt(1.0 + 4.0 * d1 - 4.0 * d1 * d1)) / 8.0;
        w[2] = (3.0 - 2.0 * d2 + std::sqrt(1.0 + 4.0 * d2 - 4.0 * d2 * d2)) / 8.0;
        w[3] = (5.0 - 2.0 * d3 - std::sqrt(-7.0 + 12.0 * d3 - 4.0 * d3 * d3)) / 8.0;
        // printf("dx : %f, d1 : %f, d2 : %f, d3 : %f\n", dx, d1, d2, d3);
        // printf("w[0] : %f, w[1] : %f, w[2] : %f, w[3] : %f\n", w[0], w[1], w[2], w[3]);
    }
    if constexpr (kernel_width == 5)
    {
        // T dx = x - base_node; // r >= 1.5 && r < 2.5
        T d1 = dx - 1.0;      // r >= 0.5 && r < 1.5
        T d2 = dx - 2.0;      // r < 0.5
        T d3 = 3.0 - dx;      // r >= 0.5 && r < 1.5
        T d4 = 4.0 - dx;      // r >= 1.5 && r < 2.5
        // TODO: not implemented.
    }
    if constexpr (kernel_width == 6){
        // T dx = x - base_node;    // r >= 2  && r < 3
        T d1 = dx - 1.0;            // r >= 1  && r < 2
        T d2 = dx - 2.0;            // r >= 0  && r < 1
        T d3 = dx - 3.0;            // r >= -1 && r <= 0
        T d4 = dx - 4.0;            // r >= -2  && r < 2
        T d5 = dx - 5.0;            // r >= -3 && r < -2

        T  alpha, beta, gamma, discr, R, R2, R3, K = 0.0;
	    int sgn;

        {
            T r = d5;
            R = r - ceil(r) + 1;  /* R between [0,1] */
            R2 = R * R; R3 = R2*R;
            alpha = 28.;
            beta  = 9./4 - 1.5 * (K + R2) + (22./3-7*K)*R - 7./3*R3;
            gamma = 0.25 * ( 0.5*(161./36 - 59./6*K + 5*K*K)*R2 + 1./3*(-109./24 + 5*K)*R2*R2 + 5./18*R3*R3  );
            discr = beta*beta - 4 * alpha * gamma;
            sgn = ((1.5-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
            w[5] = 1./(2*alpha) * ( -beta + sgn * sqrt(discr) );
        }
        {
            T r = d4;
            R = r - ceil(r) + 1;  /* R between [0,1] */
            R2 = R * R; R3 = R2*R;
            alpha = 28.;
            beta  = 9./4 - 1.5 * (K + R2) + (22./3-7*K)*R - 7./3*R3;
            gamma = 0.25 * ( 0.5*(161./36 - 59./6*K + 5*K*K)*R2 + 1./3*(-109./24 + 5*K)*R2*R2 + 5./18*R3*R3  );
            discr = beta*beta - 4 * alpha * gamma;
            sgn = ((1.5-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
            w[4] = -3./(2*alpha) * ( -beta + sgn * sqrt(discr) ) -
                1./16 + 1./8*( K+(r+2)*(r+2) ) + 1./12*(3*K-1)*(r+2) + 1./12*(r+2)*(r+2)*(r+2); 
        }{
            T r = d3;
            R = r - ceil(r) + 1;  /* R between [0,1] */
            R2 = R * R; R3 = R2*R;
            alpha = 28.;
            beta  = 9./4 - 1.5 * (K + R2) + (22./3-7*K)*R - 7./3*R3;
            gamma = 0.25 * ( 0.5*(161./36 - 59./6*K + 5*K*K)*R2 + 1./3*(-109./24 + 5*K)*R2*R2 + 5./18*R3*R3  );
            discr = beta*beta - 4 * alpha * gamma;
            sgn = ((1.5-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
            w[3] = 2./(2*alpha) * ( -beta + sgn * sqrt(discr) ) + 
                1./4 + 1./6*(4-3*K)*(r+1) - 1./6*(r+1)*(r+1)*(r+1);	
        }{      
                        T r = d2;
            R = r - ceil(r) + 1;  /* R between [0,1] */
            R2 = R * R; R3 = R2*R;
            alpha = 28.;
            beta  = 9./4 - 1.5 * (K + R2) + (22./3-7*K)*R - 7./3*R3;
            gamma = 0.25 * ( 0.5*(161./36 - 59./6*K + 5*K*K)*R2 + 1./3*(-109./24 + 5*K)*R2*R2 + 5./18*R3*R3  );
            discr = beta*beta - 4 * alpha * gamma;
            sgn = ((1.5-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
            w[2] = 2./(2*alpha) * ( -beta + sgn * sqrt(discr) ) +
                5./8 - 1./4 * ( K+r*r );
        }{
                        T r = d1;
            R = r - ceil(r) + 1;  /* R between [0,1] */
            R2 = R * R; R3 = R2*R;
            alpha = 28.;
            beta  = 9./4 - 1.5 * (K + R2) + (22./3-7*K)*R - 7./3*R3;
            gamma = 0.25 * ( 0.5*(161./36 - 59./6*K + 5*K*K)*R2 + 1./3*(-109./24 + 5*K)*R2*R2 + 5./18*R3*R3  );
            discr = beta*beta - 4 * alpha * gamma;
            sgn = ((1.5-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
            w[1] = -3./(2*alpha) * ( -beta + sgn * sqrt(discr) ) + 
                1./4 - 1./6*(4-3*K)*(r-1) + 1./6*(r-1)*(r-1)*(r-1);
        }{
            T r = dx;
            R = r - ceil(r) + 1;  /* R between [0,1] */
            R2 = R * R; R3 = R2*R;
            alpha = 28.;
            beta  = 9./4 - 1.5 * (K + R2) + (22./3-7*K)*R - 7./3*R3;
            gamma = 0.25 * ( 0.5*(161./36 - 59./6*K + 5*K*K)*R2 + 1./3*(-109./24 + 5*K)*R2*R2 + 5./18*R3*R3  );
            discr = beta*beta - 4 * alpha * gamma;
            sgn = ((1.5-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
            w[0] = 1./(2*alpha) * ( -beta + sgn * sqrt(discr) ) - 
                1./16 + 1./8*(K+(r-2)*(r-2)) - 1./12*(3*K-1)*(r-2) - 1./12*(r-2)*(r-2)*(r-2); 
        }

        
    }

}


template <typename T, size_t kernel_width>
void evaluate_kernel_bspline(const T dx, T *w, T *dw = nullptr, T *ddw = nullptr){
    if constexpr (kernel_width == 2)
    {
        w[0] = 1 - dx;
        w[1] = dx;
        if(dw){
            dw[0] = -1.0;
            dw[1] = 1.0;
        }
    }
    if constexpr (kernel_width == 3)
    {
        // T dx = x - base_node; // r >= 0.5 && r < 1.5
        T d1 = dx - 1.0;         // r < 0.5
        T d2 = 2.0 - dx;         // r >= 0.5 && r < 1.5

        T z = ((T)1.5 - dx);
        T z2 = z * z;
        w[0] = (T)0.5 * z2;
        w[1] = (T)0.75 - d1 * d1;
        T zz = (T)1.5 - d2;
        T zz2 = zz * zz;
        w[2] = (T)0.5 * zz2;

        if(dw){
            dw[0] = -z;
            dw[1] = -(T)2 * d1;
            dw[2] = zz;
        }
    }
}


template <typename T, size_t kernel_width>
void evaluate_kernel_others(const T dx, T *w, T *dw = nullptr, T *ddw = nullptr){}



// template <typename T, template <typename, size_t> class Vector, size_t kernel_width>
// void evaluate_kernel_bspline(const T x, size_t &base_node, Vector<T, kernel_width> &w)
// {
//     // 创建 base_node
//     base_node = baseNode<kernel_width>(x);
//     T dx = x - base_node;

//     static_assert(kernel_width >= 1 && kernel_width < 5, "kernel_width must >= 1");

//     // http://www.chebfun.org/examples/approx/BSplineConv.html
//     if constexpr (kernel_width == 1)
//     {
//         w.data[0] = 1;
//     }
//     if constexpr (kernel_width == 2)
//     {
//         w.data[0] = 1 - dx;
//         w.data[1] = dx;
//     }
//     if constexpr (kernel_width == 3)
//     {
//         T z = ((T)1.5 - dx);
//         T z2 = z * z;
//         w(0) = (T)0.5 * z2;
//         T d1 = dx - 1;
//         w(1) = (T)0.75 - d1 * d1;
//         T d2 = 1 - d1;
//         T zz = (T)1.5 - d2;
//         T zz2 = zz * zz;
//         w(2) = (T)0.5 * zz2;
//     }
//     if constexpr (kernel_width == 4)
//     {
//         T z = 2 - dx;
//         T z3 = z * z * z;
//         w(0) = ((T)1 / (T)6) * z3;
//         T d1 = dx - 1;
//         T zz2 = d1 * d1;
//         w(1) = ((T)0.5 * d1 - 1) * zz2 + (T)2 / (T)3;
//         T d2 = 2 - dx;
//         T zzz2 = d2 * d2;
//         w(2) = ((T)0.5 * d2 - 1) * zzz2 + (T)2 / (T)3;
//         T d3 = 3 - dx;
//         T zzzz = 2 - d3;
//         T zzzz3 = zzzz * zzzz * zzzz;
//         w(3) = ((T)1 / (T)6) * zzzz3;
//     }
// }

// template <typename T, template <typename, size_t> class Vector, size_t kernel_width>
// void evaluate_kernel_peskin(const T x, size_t &base_node, Vector<T, kernel_width> &w)
// {
//     // 创建 base_node
//     base_node = baseNode<kernel_width>(x);

//     static_assert(kernel_width == 4, "kernel_width must == 4");

//     // http://www.chebfun.org/examples/approx/BSplineConv.html
//     if constexpr (kernel_width == 1)
//     {
//         w.data[0] = 1;      // r < 0.5
//     }
//     if constexpr (kernel_width == 3)
//     {

//         T dx = x - base_node; // r >= 0.5 && r < 1.5
//         T d1 = dx - 1.0;      // r < 0.5
//         T d2 = 2.0 - dx;      // r >= 0.5 && r < 1.5

//         w(0) = (5.0 - 3.0 * dx - std::sqrt(-3.0 * (1.0 - dx) * (1.0 - dx) + 1.0)) / 6.0;
//         w(1) = (1.0 + std::sqrt(1.0 - 3.0 * d1 * d1)) / 3.0;
//         w(2) = (5.0 - 3.0 * d2 - std::sqrt(-3.0 * (1.0 - d2) * (1.0 - d2) + 1.0)) / 6.0;
//     }


// }
