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
#include "IBMesh.h"

using namespace dolfin;

template <typename GridState, typename Index, typename Particle, typename T>
class FunctorInterpolate3D
{
public:
	void operator()(GridState &grid_state, Particle &particle, const Index &base_node, T wijk, T dwijkdxi, T dwijkdxj, T dwijkdxk) const
	{
		particle.u1 += grid_state.x * wijk;
		particle.u2 += grid_state.y * wijk;
		particle.u3 += grid_state.z * wijk;
	}
};

template <typename GridState, typename Index, typename Particle, typename T>
class FunctorSpread3D
{
public:
	FunctorSpread3D(T dx, T dy, T dz) : dx(dx), dy(dy), dz(dz) {}
	T dx, dy, dz;
	void operator()(GridState &grid_state, Particle &particle, const Index &base_node, T wijk, T dwijkdxi, T dwijkdxj, T dwijkdxk) const
	{
		grid_state.x += particle.u1 * wijk * particle.w / dx / dy / dz;
		grid_state.y += particle.u2 * wijk * particle.w / dx / dy / dz;
		grid_state.z += particle.u3 * wijk * particle.w / dx / dy / dz;
	}
};

template <typename Particle, typename Grid, typename Kernel, typename Function>
void iterate_grid_3D(Grid &grid, Particle &particle, const Kernel &kernel, const Function &function)
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
			for (size_t k = 0; k < kernel_width::_2; k++)
			{
				index_type node{
					kernel.base_node[0] + i,
					kernel.base_node[1] + j,
					kernel.base_node[2] + k};

				if (node.i >= grid.grid_size.i || node.j >= grid.grid_size.j || node.k >= grid.grid_size.k)
				{
					printf("node out of range : %ld, %ld, %ld\n", node.i, node.j, node.k);
					continue;
				}
				if (node.i < 0 || node.j < 0 || node.k < 0)
				{
					printf("node out of range : %ld, %ld, %ld\n", node.i, node.j, node.k);
					continue;
				}
				auto wi = kernel.w[i];
				auto wj = kernel.w[kernel_width::_0 + j];
				auto wk = kernel.w[kernel_width::_0 + kernel_width::_1 + k];
				// printf("wi : %f, wj : %f, wk : %f\n", wi, wj, wk);
				auto wijk = wi * wj * wk;
				auto dwijkdxi = wj * wk * kernel.one_over_dh[0] * kernel.dw[i];
				auto dwijkdxj = wi * wk * kernel.one_over_dh[1] * kernel.dw[kernel_width::_0 + j];
				auto dwijkdxk = wi * wj * kernel.one_over_dh[2] * kernel.dw[kernel_width::_0 + kernel_width::_1 + k];
				state_type &grid_state = grid.get_state(node);
				function(grid_state, particle, node, wijk, dwijkdxi, dwijkdxj, dwijkdxk);
			}
		}
	}
}

class IBMesh3D
{
public:
	IBMesh3D(
		std::array<dolfin::Point, 2> points,
		std::array<size_t, 3> dims,
		size_t order)
		: order(order)
	{
		// generate mesh
		mesh_ptr = std::make_shared<Mesh>(
			BoxMesh::create(points, {dims[0], dims[1], dims[2]}, CellType::Type::hexahedron));

		nx = order * dims[0] + 1;
		ny = order * dims[1] + 1;
		nz = order * dims[2] + 1;

		x0 = points[0].x();
		x1 = points[1].x();
		y0 = points[0].y();
		y1 = points[1].y();
		z0 = points[0].z();
		z1 = points[1].z();

		dx = (x1 - x0) / static_cast<double>(dims[0]) / order;
		dy = (y1 - y0) / static_cast<double>(dims[1]) / order;
		dz = (z1 - z0) / static_cast<double>(dims[2]) / order;

		printf("order : %ld\n", order);
		printf("mesh size : %ld, %ld, %ld\n", nx, ny, nz);
		printf("mesh size : %f, %f, %f\n", dx, dy, dz);
	}
	void distribution(
		std::vector<double3> &data_to,
		const std::vector<Particle<double>> &data_from,
		const std::vector<Particle<double>> &coordinates) const
	{
		// assert(data_from.size() == coordinates.size());
		constexpr size_t dim = 3;
		constexpr size_t kernel_width_x = 4;
		constexpr size_t kernel_width_y = 4;
		constexpr size_t kernel_width_z = 4;

		using MyGrid = Grid<double3>;
		using PV = PlaceValue<octal_to_decimal<kernel_width_x, kernel_width_y, kernel_width_z>()>;
		using LKernel = IBKernel<PV, double, dim, std::array>;
		using Spread = FunctorSpread3D<MyGrid::state_type, MyGrid::index_type, Particle<double>, double>;

		MyGrid grid({nx, ny, nz});
		size_t num_lagrangian = coordinates.size();
		for (size_t idx = 0; idx < num_lagrangian; idx++)
		{
			Particle<MyGrid::value_type> particle;
			particle.x = coordinates[idx].x;
			particle.y = coordinates[idx].y;
			particle.z = coordinates[idx].z;
			particle.u1 = data_from[idx].u1;
			particle.u2 = data_from[idx].u2;
			particle.u3 = data_from[idx].u3;
			particle.w = data_from[idx].w;
			LKernel kernel({particle.x, particle.y, particle.z}, {dx, dy, dz});
			iterate_grid_3D(grid, particle, kernel, Spread(dx, dy, dz));
		}

		// 将 grid 复制到 data_to 中
		grid.copy_to(data_to);
	}

	void interpolation(
		std::vector<Particle<double>> &data_to,
		const std::vector<double3> &data_from,
		const std::vector<Particle<double>> &coordinates) const
	{
		// assert(data_to.size() == coordinates.size());
		constexpr size_t dim = 3;
		constexpr size_t kernel_width_x = 4;
		constexpr size_t kernel_width_y = 4;
		constexpr size_t kernel_width_z = 4;

		using MyGrid = Grid<double3>;
		using PV = PlaceValue<octal_to_decimal<kernel_width_x, kernel_width_y, kernel_width_z>()>;
		using LKernel = IBKernel<PV, double, dim, std::array>;
		using Interpolate = FunctorInterpolate3D<MyGrid::state_type, MyGrid::index_type, Particle<MyGrid::value_type>, MyGrid::value_type>;

		MyGrid grid({nx, ny, nz});
		// 将 data_from 复制到 grid 中
		grid.copy_from(data_from);
		size_t num_lagrangian = coordinates.size();
		data_to.resize(num_lagrangian);
		for (size_t idx = 0; idx < num_lagrangian; idx++)
		{
			Particle<MyGrid::value_type> particle;
			particle.x = coordinates[idx].x;
			particle.y = coordinates[idx].y;
			particle.z = coordinates[idx].z;
			LKernel kernel({particle.x, particle.y, particle.z}, {dx, dy, dz});
			iterate_grid_3D(grid, particle, kernel, Interpolate());
			data_to[idx].u1 = particle.u1;
			data_to[idx].u2 = particle.u2;
			data_to[idx].u3 = particle.u3;
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
				Point point(coordinates[k][0], coordinates[k][1], coordinates[k][2]);
				size_t hash = get_hash(point);
				global_map[hash] = cell_dofmap[k];
			}
		}
	}

	struct Index
	{
		size_t i, j, k;
	};
	Index get_index(const double &x, const double &y, const double &z) const
	{
		Index index{};
		index.i = static_cast<size_t>(std::round((x - x0) / dx));
		index.j = static_cast<size_t>(std::round((y - y0) / dy));
		index.k = static_cast<size_t>(std::round((z - z0) / dz));
		return index;
	}

	Index get_index(const dolfin::Point &point) const
	{
		return get_index(point.x(), point.y(), point.z());
	}

	size_t get_hash(const dolfin::Point &point) const
	{
		auto index = get_index(point);
		return index.j * nx + index.i + index.k * nx * ny;
	}

	void extract_dofs(std::vector<double3> &data, const Function &function) const
	{
		data.resize(nx * ny * nz);
		for (size_t i = 0; i < nx; i++)
		{
			for (size_t j = 0; j < ny; j++)
			{
				for (size_t k = 0; k < nz; k++)
				{
					size_t hash = i + j * nx + k * nx * ny;
					double3 tmp{};
					tmp.x = function.vector()->getitem(global_map[hash]);
					tmp.y = function.vector()->getitem(global_map[hash] + 1);
					tmp.z = function.vector()->getitem(global_map[hash] + 2);
					data[hash] = tmp;
				}
			}
		}
		// printf("extract dofs size : %ld\n", data.size());
	}

	void assign_dofs(const std::vector<double3> &data, Function &function) const
	{
		for (size_t i = 0; i < nx; i++)
		{
			for (size_t j = 0; j < ny; j++)
			{
				for (size_t k = 0; k < nz; k++)
				{

					size_t hash = i + j * nx + k * nx * ny;
					function.vector()->setitem(global_map[hash], data[hash].x);
					function.vector()->setitem(global_map[hash] + 1, data[hash].y);
					function.vector()->setitem(global_map[hash] + 2, data[hash].z);
				}
			}
		}
	}

	std::vector<double> evaluate(const Point point, const Function &function)
	{
		std::vector<double3> data_from;
		extract_dofs(data_from, function);

		std::vector<Particle<double>> data_to(1);
		std::vector<Particle<double>> coordinates(1);
		{
			coordinates[0].x = point.x();
			coordinates[0].y = point.y();
			coordinates[0].z = point.z();
		}
		interpolation(data_to, data_from, coordinates);
		auto p = data_to[0];
		return {p.u1, p.u2, p.u3};
	}

private:
	double x0, x1, y0, y1, z0, z1;
	double dx, dy, dz;
	size_t nx, ny, nz;
	size_t top_dim = 3;
	int order;

	// The map of global index to hash index for cells.
	std::vector<size_t> global_map;
	std::shared_ptr<Mesh> mesh_ptr;
};
