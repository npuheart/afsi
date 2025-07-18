/// @date 2024-07-12
/// @file IBMesh.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief 需要建立映射 (i,j) -> hash(i,j) -> dof index
///
/// The helper class for organizing the dofs on the background mesh.
///        The dofs of finite element function are organized in a 1D array.
/// @change log: 删除pybind11的头文件，将其放到IB.cpp中。

#pragma once
#include <iostream>
#include <dolfin.h>

#include "spatial/kernel_helper.h"
#include "spatial/kernel_expression.h"
#include "spatial/metegrid.h"

using namespace dolfin;

struct double2
{
	static constexpr std::size_t _dim = 2;
	using value_type = double;
	double x, y, z;
	double u1, u2, u3;
	double du1[3] = {};
	double du2[3] = {};
	double du3[3] = {};
	double2()
		: x{}, y{}, z{}, u1{}, u2{}, u3{}
	{
	}
	double2(double x, double y)
		: x{x}, y{y}, z{}, u1{}, u2{}, u3{}
	{
	}
	double2(double x, double y, double z)
		: x{x}, y{y}, z{z}, u1{}, u2{}, u3{}
	{
	}
};

struct double3
{
	static constexpr std::size_t _dim = 3;
	using value_type = double;
	double x, y, z;
	double u1, u2, u3;
	double du1[3] = {};
	double du2[3] = {};
	double du3[3] = {};
	double3()
		: x{}, y{}, z{}, u1{}, u2{}, u3{}
	{
	}
	double3(double x, double y)
		: x{x}, y{y}, z{}, u1{}, u2{}, u3{}
	{
	}
	double3(double x, double y, double z)
		: x{x}, y{y}, z{z}, u1{}, u2{}, u3{}
	{
	}
};

template <typename T>
struct Particle
{
	T x, y, z;
	T w;
	T u1, u2, u3;
	Particle()
		: x{}, y{}, z{}, w{}, u1{}, u2{}, u3{} // Initializes all members to their default values
	{
	}
};

template <typename GridState, typename Index, typename Particle, typename T>
class FunctorInterpolate
{
public:
	void operator()(GridState &grid_state, Particle &particle, const Index &base_node, T wij, T dwijdxi, T dwijdxj) const
	{
		particle.u1 += grid_state.x * wij;
		particle.u2 += grid_state.y * wij;
	}
};

template <typename GridState, typename Index, typename Particle, typename T>
class FunctorSpread
{
public:
	FunctorSpread(T dx, T dy) : dx(dx), dy(dy) {}
	T dx, dy;
	void operator()(GridState &grid_state, Particle &particle, const Index &base_node, T wij, T dwijdxi, T dwijdxj) const
	{
		grid_state.x += particle.u1 * wij * particle.w / dx / dy;
		grid_state.y += particle.u2 * wij * particle.w / dx / dy;
	}
};

template <typename Particle, typename Grid, typename Kernel, typename Function>
void iterate_grid_2D(Grid &grid, Particle &particle, const Kernel &kernel, const Function &function)
{
	using index_type = typename Grid::index_type;
	using state_type = typename Grid::state_type;
	using value_type = typename Grid::value_type;
	using kernel_width = typename Kernel::kernel_width;

	value_type{}; // 阻止编译器警告。

	for (size_t i = 0; i < kernel_width::_0; i++)
	{
		for (size_t j = 0; j < kernel_width::_1; j++)
		{
			index_type node{kernel.base_node[0] + i, kernel.base_node[1] + j};
			if (node.i >= grid.grid_size.i || node.j >= grid.grid_size.j)
			{
				// printf("node out of range : %ld, %ld\n", node.i, node.j);
				continue;
			}
			if (node.i < 0 || node.j < 0)
			{
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

class IBMesh
{
public:
	IBMesh(
		std::array<dolfin::Point, 2> points,
		std::array<size_t, 2> dims,
		size_t order)
		: order(order)
	{
		// generate mesh
		mesh_ptr = std::make_shared<Mesh>(
			RectangleMesh::create(points, {dims[0], dims[1]}, CellType::Type::quadrilateral));

		nx = order * dims[0] + 1;
		ny = order * dims[1] + 1;

		x0 = points[0].x();
		x1 = points[1].x();
		y0 = points[0].y();
		y1 = points[1].y();

		dx = (x1 - x0) / static_cast<double>(dims[0]) / order;
		dy = (y1 - y0) / static_cast<double>(dims[1]) / order;

		printf("order : %ld\n", order);
		printf("mesh size : %ld, %ld\n", nx, ny);
		printf("mesh size : %f, %f\n", dx, dy);
	}
	void distribution(
		std::vector<double2> &data_to,
		const std::vector<Particle<double>> &data_from,
		const std::vector<Particle<double>> &coordinates) const
	{
		// assert(data_from.size() == coordinates.size());
		constexpr size_t dim = 2;
		constexpr size_t kernel_width_x = 4;
		constexpr size_t kernel_width_y = 4;

		using MyGrid = Grid<double2>;
		using PV = PlaceValue<octal_to_decimal<kernel_width_x, kernel_width_y>()>;
		using LKernel = IBKernel<PV, double, dim, std::array>;
		using Spread = FunctorSpread<MyGrid::state_type, MyGrid::index_type, Particle<double>, double>;

		MyGrid grid({nx, ny});
		size_t num_lagrangian = coordinates.size();
		for (size_t idx = 0; idx < num_lagrangian; idx++)
		{
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

	void interpolation(
		std::vector<Particle<double>> &data_to,
		const std::vector<double2> &data_from,
		const std::vector<Particle<double>> &coordinates) const
	{
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
		for (size_t idx = 0; idx < num_lagrangian; idx++)
		{
			Particle<MyGrid::value_type> particle;
			particle.x = coordinates[idx].x;
			particle.y = coordinates[idx].y;
			LKernel kernel({particle.x, particle.y}, {dx, dy});
			iterate_grid_2D(grid, particle, kernel, Interpolate());
			data_to[idx].u1 = particle.u1;
			data_to[idx].u2 = particle.u2;
		}
	}
	std::shared_ptr<Mesh> mesh()
	{
		return mesh_ptr;
	}

	void build_map(const Function &coordinates)
	{
		// The local map local to global
		std::vector<size_t> local_map;

		// TODO : check if mesh is the same as mesh_ptr
		auto mesh = coordinates.function_space()->mesh();
		auto dofmap = coordinates.function_space()->dofmap();

		/// Get The element of function space
		auto element = coordinates.function_space()->element();
		auto value_size = coordinates.value_size();
		auto global_size = coordinates.function_space()->dim();
		global_map.resize(global_size / value_size);
		/// Get local to global dofmap
		std::vector<size_t> local_to_global;
		dofmap->tabulate_local_to_global_dofs(local_to_global);

		for (dolfin::CellIterator e(*mesh_ptr); !e.end(); ++e)
		{
			// Step 1 : get coordinates of cell dofs
			Cell cell(*mesh, e->global_index());
			std::vector<double> coordinate_dofs;
			cell.get_coordinate_dofs(coordinate_dofs);
			boost::multi_array<double, 2> coordinates;
			element->tabulate_dof_coordinates(coordinates, coordinate_dofs, cell);

			// Step 2 : get the dof map
			auto cell_dofmap = dofmap->cell_dofs(cell.index());

			// Step 3 : iterate dof_coordinates of the cell
			for (size_t k = 0; k < cell_dofmap.size() / value_size; k++)
			{
				Point point(coordinates[k][0], coordinates[k][1]);
				size_t hash = get_hash(point);
				global_map[hash] = cell_dofmap[k];
			}
		}
	}

	struct Index
	{
		size_t i, j;
	};
	Index get_index(const double &x, const double &y) const
	{
		Index index{};
		index.i = static_cast<size_t>(std::round((x - x0) / dx));
		index.j = static_cast<size_t>(std::round((y - y0) / dy));
		return index;
	}

	Index get_index(const dolfin::Point &point) const
	{
		return get_index(point.x(), point.y());
	}

	size_t get_hash(const dolfin::Point &point) const
	{
		auto index = get_index(point);
		return index.j * nx + index.i;
	}

	void extract_dofs(std::vector<double2> &data, const Function &function) const
	{
		data.resize(nx * ny);
		for (size_t i = 0; i < nx; i++)
		{
			for (size_t j = 0; j < ny; j++)
			{
				size_t hash = i + j * nx;
				double2 tmp{};
				tmp.x = function.vector()->getitem(global_map[hash]);
				tmp.y = function.vector()->getitem(global_map[hash] + 1);
				data[hash] = tmp;
			}
		}
		// printf("extract dofs size : %ld\n", data.size());
	}

	void assign_dofs(const std::vector<double2> &data, Function &function) const
	{
		for (size_t i = 0; i < nx; i++)
		{
			for (size_t j = 0; j < ny; j++)
			{
				size_t hash = i + j * nx;

				function.vector()->setitem(global_map[hash], data[hash].x);
				function.vector()->setitem(global_map[hash] + 1, data[hash].y);
			}
		}
	}

	std::vector<double> evaluate(const Point point, const Function &function)
	{
		std::vector<double2> data_from;
		extract_dofs(data_from, function);

		std::vector<Particle<double>> data_to(1);
		std::vector<Particle<double>> coordinates(1);
		{
			coordinates[0].x = point.x();
			coordinates[0].y = point.y();
		}
		interpolation(data_to, data_from, coordinates);
		auto p = data_to[0];
		return {p.u1, p.u2};
	}

private:
	double x0, x1, y0, y1;
	double dx, dy;
	size_t nx, ny;
	size_t top_dim = 2;
	int order;

	// The map of global index to hash index for cells.
	std::vector<size_t> global_map;
	std::shared_ptr<Mesh> mesh_ptr;
};
