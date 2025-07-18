#include "QuadratureRules.h"
#include <array>
#include <iostream>

void linearInterpolation(double* result, const std::array<double, 2> a, const std::array<double, 2> b, double alpha) {
    for (std::size_t i = 0; i < 2; ++i) {
        result[i] = (1 - alpha) * a[i] + alpha * b[i];
    }
}

void linearInterpolation(double* result, const std::array<double, 3> a, const std::array<double, 3> b,
                         const std::array<double, 3> c, double alpha, double beta) {
    for (std::size_t i = 0; i < 3; ++i) {
        result[i] = (1 - alpha - beta) * a[i] + alpha * b[i] + beta * c[i];
    }
}

const size_t num_GQ_points = 3;

void main_1(double* ab, double* ac, double* bc) {
//     double GQ_points[]  = {0.1127016653792583, 0.5, 0.8872983346207417};
//     double GQ_weights[] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};

//     double a[] = {0.0, 0.0};
//     double b[] = {0.0, 1.0};
//     double c[] = {1.0, 0.0};

//     for (size_t i = 0; i < num_GQ_points; i++) {
//         double result[2];
//         linearInterpolation<2>(result, c, b, GQ_points[i]);
//         std::cout << result[0] << " " << result[1] << std::endl;
//     }

//     size_t line[2]     = {0, 2};
//     size_t triangle[3] = {0, 1, 2};
//     double local_GQ_points[DIM * num_GQ_points];

//     if (line[0] + line[1] == triangle[0] + triangle[1]) { return ab; }
//     if (line[0] + line[1] == triangle[0] + triangle[2]) { return ac; }
//     if (line[0] + line[1] == triangle[1] + triangle[2]) { return bc; }

}

template <typename TV, typename T, int order, int dim>
struct SurfaceSimplexQuadrature {

    const static npuheart::SimplexQuadrature<TV, T, order, dim>     cell;
    const static npuheart::SimplexQuadrature<TV, T, order, dim - 1> facet;

    double surface_gauss_points[dim + 1][facet.num_points() * dim];

    SurfaceSimplexQuadrature() {
        // 计算边界的高斯积分点
        if constexpr (dim == 2) {
            fun(surface_gauss_points[2], {0.0, 0.0}, {0.0, 1.0});
            fun(surface_gauss_points[1], {0.0, 0.0}, {1.0, 0.0});
            fun(surface_gauss_points[0], {0.0, 1.0}, {1.0, 0.0});
        } else if constexpr (dim == 3) {
            fun(surface_gauss_points[0], {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});
            fun(surface_gauss_points[1], {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0});
            fun(surface_gauss_points[2], {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0});
            fun(surface_gauss_points[3], {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0});
        } else {
            static_assert(dim == 2 || dim == 3, "dim must be 2 or 3");
        }
    }

    void fun(double* surface_gauss_points, const std::array<double, 2> a, const std::array<double, 2> b) {
        for (size_t i = 0; i < facet.num_points(); i++) {
            linearInterpolation(&surface_gauss_points[dim * i], a, b, facet._points[i]._x);
        }
    }

    void fun(double* surface_gauss_points, const std::array<double, 3> a, const std::array<double, 3> b, const
    std::array<double, 3> c) {
        for (size_t i = 0; i < facet.num_points(); i++) {
            linearInterpolation(&surface_gauss_points[dim * i], a, b, c, facet._points[i]._x,  facet._points[i]._y);
        }
    }

    const double* select_surface_gauss_points(const size_t i) {
        return surface_gauss_points[i];
        return 0;
    }
};

template <typename TV, typename T, int order, int dim>
const npuheart::SimplexQuadrature<TV, T, order, dim> SurfaceSimplexQuadrature<TV, T, order, dim>::cell;

template <typename TV, typename T, int order, int dim>
const npuheart::SimplexQuadrature<TV, T, order, dim - 1> SurfaceSimplexQuadrature<TV, T, order, dim>::facet;

const size_t GQ_ORDER = 5;
const size_t DIM      = 1;

// 计算规范面积坐标系的高斯积分点
// 二维平面上的任意一点可以表示为 $G = \alpha A + \beta B$。如果 $\alpha + \beta = 1$，
// 则点 $G$ 落在直线 $AB$ 上。例如，当 $A = (0, 0)$，$B = (1, 0)$，$G = (g, 0)$ 时，
// 有 $G' = (1 - g)A' + gB'$。

// 同样，三维空间中的任意一点可以表示为 $G = \alpha A + \beta B + \gamma C$。如果
// $\alpha + \beta + \gamma = 1$，则点 $G$ 落在由 $ABC$ 表示的平面上。以参考三角
// 形 $A = (0, 0, 0)$，$B = (0, 1, 0)$，$C = (1, 0, 0)$ 和 $G = (g_x, g_y, 0)$
// 为例，有 $G' = (1 - g_x - g_y)A' + g_xB' + g_yC'$。


// 对于三角形cell=[A,B,C]，local_facet_id = 0, 1, 2 分别表示边
// BC=[cell[1], cell[2]], 
// AC=[cell[0], cell[2]], 
// AB=[cell[0], cell[1]], 
// 即对面的边。

int main_1() {


    auto a = npuheart::simplex_quadrature<npuheart::double3, double, GQ_ORDER, DIM>;
    printf("高斯积分点数量: %d\n", a.num_points());
    printf("高斯积分阶数  : %ld\n", GQ_ORDER);
    for (size_t i = 0; i < a.num_points(); i++) {
        printf("高斯积分点    : %f %f %f\n", a._points[i]._x, a._points[i]._y, a._points[i]._z);
    }

    auto b = npuheart::simplex_quadrature<npuheart::double3, double, 3, 2>;
    printf("高斯积分点数量: %d\n", b.num_points());
    printf("高斯积分阶数  : %ld\n", GQ_ORDER);
    for (size_t i = 0; i < b.num_points(); i++) {
        printf("高斯积分点    : %f %f %f\n", b._points[i]._x, b._points[i]._y, b._points[i]._z);
    }

    auto c = npuheart::simplex_quadrature<npuheart::double3, double, GQ_ORDER, 3>;
    printf("高斯积分点数量: %d\n", c.num_points());
    printf("高斯积分阶数  : %ld\n", GQ_ORDER);
    for (size_t i = 0; i < c.num_points(); i++) {
        printf("高斯积分点    : %f %f %f\n", c._points[i]._x, c._points[i]._y, c._points[i]._z);
    }

    SurfaceSimplexQuadrature<npuheart::double3, double, GQ_ORDER, 2> surface_2d;
    for (size_t j = 0; j < 3; j++) {
        for (size_t i = 0; i < num_GQ_points; i++) {
            printf("高斯积分点    : %f %f\n", surface_2d.surface_gauss_points[j][2 * i],
                   surface_2d.surface_gauss_points[j][2 * i + 1]);
        }
    }

    SurfaceSimplexQuadrature<npuheart::double3, double, GQ_ORDER, 3> surface_3d;
    for (size_t j = 0; j < 4; j++) {
        for (size_t i = 0; i < num_GQ_points; i++) {
            printf("高斯积分点    : %f %f %f\n", surface_3d.surface_gauss_points[j][3 * i],
                   surface_3d.surface_gauss_points[j][3 * i + 1], surface_3d.surface_gauss_points[j][3 * i + 2]);
        }
    }

    return 0;
}