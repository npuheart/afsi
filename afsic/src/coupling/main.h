#pragma once
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>

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
        for (size_t i = 0; i < num_dofs; ++i) {
            size_t hash = get_hash(coords[i * top_dim], coords[i * top_dim + 1]);
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

  private:
    double x0, x1, y0, y1;
    double dx, dy;
    std::int64_t nx, ny;
    uint top_dim = 2;
    uint order;

    // The map of global index to hash index for cells.
    std::vector<size_t> global_map;
    std::shared_ptr<mesh::Mesh<U>> mesh_ptr;
};

int coupling();

} // namespace coupling
