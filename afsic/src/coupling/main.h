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
  IBMesh(double x0, double  x1, double y0, double y1,
         std::int64_t dim_x, std::int64_t dim_y, uint order)
      : x0(x0), x1(x1), y0(y0), y1(y1), order(order) {

    // x0 = points[0][0];
    // y0 = points[0][1];
    // x1 = points[1][0];
    // y1 = points[1][1];

    nx = order * dim_x + 1;
    ny = order * dim_y + 1;

    dx = (x1 - x0) / (nx - 1);
    dy = (y1 - y0) / (ny - 1);

    printf("order : %ld\n", order);
    printf("mesh size : %ld, %ld\n", nx, ny);
    printf("mesh size : %f, %f\n", dx, dy);

    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        MPI_COMM_WORLD, {{{x0, y0}, {x1, y1}}}, {nx, ny},
        mesh::CellType::quadrilateral, part));
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


//   std::string create_rectangle("create_rectangle_" + type);
//   m.def(
//       create_rectangle.c_str(),
//       [](MPICommWrapper comm, std::array<std::array<T, 2>, 2> p,
//          std::array<std::int64_t, 2> n, dolfinx::mesh::CellType celltype,
//          const part::impl::PythonCellPartitionFunction& part,
//          dolfinx::mesh::DiagonalType diagonal)
//       {
//         return dolfinx::mesh::create_rectangle<T>(
//             comm.get(), p, n, celltype,
//             part::impl::create_cell_partitioner_cpp(part), diagonal);
//       },
//       nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
//       nb::arg("partitioner").none(), nb::arg("diagonal"));

int coupling();

} // namespace coupling
