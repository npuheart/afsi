#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <iostream>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;
namespace coupling {

class IBMesh {
  IBMesh(std::array<dolfin::Point, 2> points, std::array<size_t, 2> dims,
         size_t order)
      : order(order) {}

      private:
	double x0, x1, y0, y1;
	double dx, dy;
	size_t nx, ny;
	size_t top_dim = 2;
	int order;

	// The map of global index to hash index for cells.
	std::vector<size_t> global_map;
	std::shared_ptr<mesh::Mesh<U>> mesh_ptr;
};

int coupling() {
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<U>>(
      mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                {32, 16}, mesh::CellType::triangle, part));

  printf("a\n");
  return 0;
}

} // namespace coupling
