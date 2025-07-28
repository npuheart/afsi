#pragma once

#include "main.h"

namespace coupling {
struct IBMesh3D {
    IBMesh3D(double x0, double x1, double y0, double y1, double z0, double z1, std::int64_t dim_x, std::int64_t dim_y, std::int64_t dim_z, uint order)
        : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1), order(order) {

        nx = order * dim_x + 1;
        ny = order * dim_y + 1;
        nz = order * dim_z + 1;

        dx = (x1 - x0) / (nx - 1);
        dy = (y1 - y0) / (ny - 1);
        dz = (z1 - z0) / (nz - 1);

        printf("order : %d\n", order);
        printf("mesh size : %ld, %ld, %ld\n", nx, ny, nz);
        printf("cell size : %f, %f, %f\n", dx, dy, dz);

        auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
        mesh_ptr = std::make_shared<mesh::Mesh<U>>(
            mesh::create_box<U>(MPI_COMM_WORLD, {{{x0, y0, z0}, {x1, y1, z1}}}, {nx, ny, nz}, mesh::CellType::hexahedron, part));
    }

    // void build_map(){
    void build_map(const std::vector<double> &coords) {
        size_t num_dofs = coords.size() / top_dim;
        global_map.resize(num_dofs);
        // printf("num_dofs : %ld\n", num_dofs);
        for (size_t i = 0; i < num_dofs; ++i) {
            size_t hash = get_hash(coords[i * top_dim], coords[i * top_dim + 1], coords[i * top_dim + 2]);
            global_map[hash] = i;
        }
    }

    const std::shared_ptr<mesh::Mesh<U>> &mesh() { return mesh_ptr; }

    struct Index {
        size_t i, j, k;
    };
    Index get_index(const double &x, const double &y, const double &z) const {
        return Index{static_cast<size_t>(std::round((x - x0) / dx)), static_cast<size_t>(std::round((y - y0) / dy)),
                     static_cast<size_t>(std::round((z - z0) / dz))};
    }

    size_t get_hash(const double &x, const double &y, const double &z) const {
        auto index = get_index(x, y, z);
        return index.k * nx * ny + index.j * nx + index.i;
    }

    void extract_dofs(std::vector<double3> &data, const std::vector<double> &function) const {
        data.resize(nx * ny);
        for (size_t i = 0; i < nx; i++) {
            for (size_t j = 0; j < ny; j++) {
                for (size_t k = 0; k < nz; k++) {
                    size_t hash = i + j * nx + k * nx * ny;
                    double3 tmp{};
                    tmp.x = function[top_dim * global_map[hash]];
                    tmp.y = function[top_dim * global_map[hash] + 1];
                    tmp.z = function[top_dim * global_map[hash] + 2];
                    data[hash] = tmp;
                }
            }
        }
    }

    void assign_dofs(const std::vector<double3> &data, std::vector<double> &function) const {

        for (size_t i = 0; i < nx; i++) {
            for (size_t j = 0; j < ny; j++) {
                for (size_t k = 0; k < nz; k++) {
                    size_t hash = i + j * nx + k * nx * ny;

                    function[top_dim * global_map[hash]] = data[hash].x;
                    function[top_dim * global_map[hash] + 1] = data[hash].y;
                    function[top_dim * global_map[hash] + 2] = data[hash].z;
                }
            }
        }
    }

    std::vector<double> evaluate(double x, double y, double z, const std::vector<double> &function) {
        std::vector<double3> data_from;
        extract_dofs(data_from, function);

        std::vector<Particle<double>> data_to(1);
        std::vector<Particle<double>> coordinates{{x, y, z}};
        interpolation(data_to, data_from, coordinates);
        auto p = data_to[0];
        return {p.u1, p.u2, p.u3};
    }

    void distribution(std::vector<double3> &data_to, const std::vector<Particle<double>> &data_from, const std::vector<Particle<double>> &coordinates) const {
        // assert(data_from.size() == coordinates.size());
        constexpr size_t dim = 3;
        constexpr size_t kernel_width_x = 4;
        constexpr size_t kernel_width_y = 4;
        constexpr size_t kernel_width_z = 4;

        using MyGrid = Grid<double3>;
        using PV = PlaceValue<octal_to_decimal<kernel_width_x, kernel_width_y, kernel_width_z>()>;
        using LKernel = IBKernel<PV, double, dim, std::array>;
        using Spread = FunctorSpread<MyGrid::state_type, MyGrid::index_type, Particle<double>, double>;

        MyGrid grid({nx, ny, nz});
        size_t num_lagrangian = coordinates.size();
        for (size_t idx = 0; idx < num_lagrangian; idx++) {
            Particle<MyGrid::value_type> particle;
            particle.x = coordinates[idx].x;
            particle.y = coordinates[idx].y;
            particle.u1 = data_from[idx].u1;
            particle.u2 = data_from[idx].u2;
            particle.w = data_from[idx].w;
            LKernel kernel({particle.x, particle.y}, {dx, dy});
            iterate_grid_2D(grid, particle, kernel, Spread(dx, dy));
        }

        // 将 grid 复制到 data_to 中
        grid.copy_to(data_to);
    }

    void interpolation(std::vector<Particle<double>> &data_to, const std::vector<double3> &data_from, const std::vector<Particle<double>> &coordinates) const {
        // assert(data_to.size() == coordinates.size());
        constexpr size_t dim = 2;
        constexpr size_t kernel_width_x = 4;
        constexpr size_t kernel_width_y = 4;

        using MyGrid = Grid<double3>;
        using PV = PlaceValue<octal_to_decimal<kernel_width_x, kernel_width_y>()>;
        using LKernel = IBKernel<PV, double, dim, std::array>;
        using Interpolate = FunctorInterpolate3D<MyGrid::state_type, MyGrid::index_type, Particle<MyGrid::value_type>, MyGrid::value_type>;

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
    uint top_dim = 3;

  private:
    double x0, x1, y0, y1, z0, z1;
    double dx, dy, dz;
    std::int64_t nx, ny, nz;
    uint order;

    // The map of global index to hash index for cells.
    std::vector<size_t> global_map;
    std::shared_ptr<mesh::Mesh<U>> mesh_ptr;
};

} // namespace coupling
