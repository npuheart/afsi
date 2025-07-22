#pragma once
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>

#include "spatial/kernel_expression.h"
#include "spatial/kernel_helper.h"
#include "spatial/metegrid.h"

struct double2 {
    static constexpr std::size_t _dim = 2;
    using value_type = double;
    double x, y, z;
    double u1, u2, u3;
    double du1[3] = {};
    double du2[3] = {};
    double du3[3] = {};
    double2() : x{}, y{}, z{}, u1{}, u2{}, u3{} {}
    double2(double x, double y) : x{x}, y{y}, z{}, u1{}, u2{}, u3{} {}
    double2(double x, double y, double z) : x{x}, y{y}, z{z}, u1{}, u2{}, u3{} {}
};

struct double3 {
    static constexpr std::size_t _dim = 3;
    using value_type = double;
    double x, y, z;
    double u1, u2, u3;
    double du1[3] = {};
    double du2[3] = {};
    double du3[3] = {};
    double3() : x{}, y{}, z{}, u1{}, u2{}, u3{} {}
    double3(double x, double y) : x{x}, y{y}, z{}, u1{}, u2{}, u3{} {}
    double3(double x, double y, double z) : x{x}, y{y}, z{z}, u1{}, u2{}, u3{} {}
};

template <typename T> struct Particle {
    T x, y, z;
    T w;
    T u1, u2, u3;
    Particle()
        : x{}, y{}, z{}, w{}, u1{}, u2{}, u3{} // Initializes all members to their default values
    {}

    Particle(T x_, T y_) : x(x_), y(y_), z{}, w{}, u1{}, u2{}, u3{} {}
};

template <typename GridState, typename Index, typename Particle, typename T> class FunctorInterpolate {
  public:
    void operator()(GridState &grid_state, Particle &particle, const Index &base_node, T wij, T dwijdxi, T dwijdxj) const {
        particle.u1 += grid_state.x * wij;
        particle.u2 += grid_state.y * wij;
    }
};

template <typename GridState, typename Index, typename Particle, typename T> class FunctorSpread {
  public:
    FunctorSpread(T dx, T dy) : dx(dx), dy(dy) {}
    T dx, dy;
    void operator()(GridState &grid_state, Particle &particle, const Index &base_node, T wij, T dwijdxi, T dwijdxj) const {
        grid_state.x += particle.u1 * wij * particle.w / dx / dy;
        grid_state.y += particle.u2 * wij * particle.w / dx / dy;
    }
};

template <typename Particle, typename Grid, typename Kernel, typename Function>
void iterate_grid_2D(Grid &grid, Particle &particle, const Kernel &kernel, const Function &function) {
    using index_type = typename Grid::index_type;
    using state_type = typename Grid::state_type;
    using value_type = typename Grid::value_type;
    using kernel_width = typename Kernel::kernel_width;

    value_type{}; // 阻止编译器警告。

    for (size_t i = 0; i < kernel_width::_0; i++) {
        for (size_t j = 0; j < kernel_width::_1; j++) {
            index_type node{kernel.base_node[0] + i, kernel.base_node[1] + j};
            if (node.i >= grid.grid_size.i || node.j >= grid.grid_size.j) {
                // printf("node out of range : %ld, %ld\n", node.i, node.j);
                continue;
            }
            if (node.i < 0 || node.j < 0) {
                // printf("node out of range : %ld, %ld\n", node.i, node.j);
                continue;
            }
            auto wi = kernel.w[i];
            auto wj = kernel.w[kernel_width::_0 + j];
            auto wij = wi * wj;
            auto dwijdxi = wj * kernel.one_over_dh[0] * kernel.dw[i];
            auto dwijdxj = wi * kernel.one_over_dh[1] * kernel.dw[kernel_width::_0 + j];
            state_type &grid_state = grid.get_state(node);
            function(grid_state, particle, node, wij, dwijdxi, dwijdxj);
        }
    }
}
using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

namespace coupling {
struct IBMesh {
    IBMesh(double x0, double x1, double y0, double y1, std::int64_t dim_x, std::int64_t dim_y, uint order) : x0(x0), x1(x1), y0(y0), y1(y1), order(order) {

        nx = order * dim_x + 1;
        ny = order * dim_y + 1;

        dx = (x1 - x0) / (nx - 1);
        dy = (y1 - y0) / (ny - 1);

        printf("order : %d\n", order);
        printf("mesh size : %ld, %ld\n", nx, ny);
        printf("cell size : %f, %f\n", dx, dy);

        auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
        mesh_ptr =
            std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{x0, y0}, {x1, y1}}}, {nx, ny}, mesh::CellType::quadrilateral, part));
    }

    // void build_map(){
    void build_map(const std::vector<double> &coords) {
        size_t num_dofs = coords.size() / top_dim;
        global_map.resize(num_dofs);
        // printf("num_dofs : %ld\n", num_dofs);
        for (size_t i = 0; i < num_dofs; ++i) {
            size_t hash = get_hash(coords[i * top_dim], coords[i * top_dim + 1]);
            // printf("hash : %ld, i : %ld, x : %f, y : %f\n", hash, i, coords[i * top_dim], coords[i * top_dim + 1]);
            global_map[hash] = i;
        }
    }

    const std::shared_ptr<mesh::Mesh<U>> &mesh() { return mesh_ptr; }

    struct Index {
        size_t i, j;
    };
    Index get_index(const double &x, const double &y) const {
        return Index{static_cast<size_t>(std::round((x - x0) / dx)), static_cast<size_t>(std::round((y - y0) / dy))};
    }

    size_t get_hash(const double &x, const double &y) const {
        auto index = get_index(x, y);
        return index.j * nx + index.i;
    }

    void extract_dofs(std::vector<double2> &data, const std::vector<double> &function) const {
        data.resize(nx * ny);
        for (size_t i = 0; i < nx; i++) {
            for (size_t j = 0; j < ny; j++) {
                size_t hash = i + j * nx;
                double2 tmp{};
                tmp.x = function[top_dim * global_map[hash]];
                tmp.y = function[top_dim * global_map[hash] + 1];
                data[hash] = tmp;
            }
        }
    }

    void assign_dofs(const std::vector<double2> &data, std::vector<double> &function) const {
        for (size_t i = 0; i < nx; i++) {
            for (size_t j = 0; j < ny; j++) {
                size_t hash = i + j * nx;

                function[top_dim * global_map[hash]] = data[hash].x;
                function[top_dim * global_map[hash] + 1] = data[hash].y;
            }
        }
    }

    std::vector<double> evaluate(double x, double y, const std::vector<double> &function) {
        std::vector<double2> data_from;
        extract_dofs(data_from, function);

        std::vector<Particle<double>> data_to(1);
        std::vector<Particle<double>> coordinates{{x, y}};
        interpolation(data_to, data_from, coordinates);
        auto p = data_to[0];
        return {p.u1, p.u2};
    }
    void interpolation(std::vector<Particle<double>> &data_to, const std::vector<double2> &data_from, const std::vector<Particle<double>> &coordinates) const {
        // assert(data_to.size() == coordinates.size());
        constexpr size_t dim = 2;
        constexpr size_t kernel_width_x = 4;
        constexpr size_t kernel_width_y = 4;

        using MyGrid = Grid<double2>;
        using PV = PlaceValue<octal_to_decimal<kernel_width_x, kernel_width_y>()>;
        using LKernel = IBKernel<PV, double, dim, std::array>;
        using Interpolate = FunctorInterpolate<MyGrid::state_type, MyGrid::index_type, Particle<MyGrid::value_type>, MyGrid::value_type>;

        MyGrid grid({nx, ny});
        // 将 data_from 复制到 grid 中
        grid.copy_from(data_from);
        size_t num_lagrangian = coordinates.size();
        data_to.resize(num_lagrangian);
        for (size_t idx = 0; idx < num_lagrangian; idx++) {
            Particle<MyGrid::value_type> particle;
            particle.x = coordinates[idx].x;
            particle.y = coordinates[idx].y;
            LKernel kernel({particle.x, particle.y}, {dx, dy});
            iterate_grid_2D(grid, particle, kernel, Interpolate());
            data_to[idx].u1 = particle.u1;
            data_to[idx].u2 = particle.u2;
        }
    }
    uint top_dim = 2;

  private:
    double x0, x1, y0, y1;
    double dx, dy;
    std::int64_t nx, ny;
    uint order;

    // The map of global index to hash index for cells.
    std::vector<size_t> global_map;
    std::shared_ptr<mesh::Mesh<U>> mesh_ptr;
};

int coupling();

class IBInterpolation {
  public:
    IBMesh &fluid_mesh;
    std::vector<Particle<double>> current_coordinates;

    IBInterpolation(IBMesh &fluid_mesh) : fluid_mesh(fluid_mesh) {}

    void evaluate_current_points(const std::vector<double> &position) {
        // 只修改 Particle 的x y 而不是 u1 u2
        assign_positions(current_coordinates, position);
    }

    void assign(std::vector<double> &position, const std::vector<Particle<double>> &data) {
        size_t value_size = fluid_mesh.top_dim;
        size_t dof_size = position.size();
        // TODO: assert(dof_size / value_size == data.size());

        for (size_t i = 0; i < dof_size / value_size; i++) {
            position[i * value_size] = data[i].u1;
            position[i * value_size + 1] = data[i].u2;
        }
    }

    void assign(std::vector<Particle<double>> &data, const std::vector<double> &position) {

        size_t value_size = fluid_mesh.top_dim;
        size_t dof_size = position.size();
        data.resize(dof_size / value_size);
        // TODO: assert(dof_size / value_size == data.size());

        for (size_t i = 0; i < data.size(); i++) {
            data[i].u1 = position[i * value_size];
            data[i].u2 = position[i * value_size + 1];
        }
    }

    void assign_positions(std::vector<Particle<double>> &data, const std::vector<double> &position) {

        size_t value_size = fluid_mesh.top_dim;
        size_t dof_size = position.size();
        data.resize(dof_size / value_size);
        // TODO: assert(dof_size / value_size == data.size());

        for (size_t i = 0; i < data.size(); i++) {
            data[i].x = position[i * value_size];
            data[i].y = position[i * value_size + 1];
        }
    }

    void fluid_to_solid(const std::vector<double> &fluid, std::vector<double> &solid) {

        std::vector<Particle<double>> array_solid;
        std::vector<double2> array_fluid;
        fluid_mesh.extract_dofs(array_fluid, fluid);
        fluid_mesh.interpolation(array_solid, array_fluid, current_coordinates);
        assign(solid, array_solid);
    }

    // void solid_to_fluid(Function &fluid, const Function &solid)
    // {
    // 	std::vector<Particle<double>> array_solid;
    // 	std::vector<double2> array_fluid;
    // 	assign(array_solid, solid);
    // 	for (size_t i = 0; i < array_solid.size(); i++)
    // 	{
    // 		array_solid[i].w = 1.0;
    // 	}
    // 	fluid_mesh->distribution(array_fluid, array_solid, current_coordinates);
    // 	fluid_mesh->assign_dofs(array_fluid, fluid);
    // }
};

} // namespace coupling
