/// @date 2024-07-12
/// @file IBInterpolation.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#pragma once
#include <numeric>
#include "IBMesh.h"
#include "IBMesh3D.h"
using namespace dolfin;

class IBInterpolation
{
public:
	std::shared_ptr<IBMesh> fluid_mesh;
	std::shared_ptr<Mesh> solid_mesh;
	std::vector<Particle<double>> current_coordinates;

	IBInterpolation(std::shared_ptr<IBMesh> fluid_mesh,
					std::shared_ptr<Mesh> solid_mesh) : fluid_mesh(fluid_mesh),
														solid_mesh(solid_mesh)
	{
	}

	void evaluate_current_points(const Function& position)
	{
		assign_coordinates(current_coordinates, position);
	}

	void assign(Function &position, const std::vector<Particle<double>>&data){
		//TODO: check value_size == 2
		size_t value_size = position.value_size();        
		size_t dof_size = position.function_space()->dim();
		std::vector<double> data_flatten(dof_size);

		// unpack the data_flatten
		for (size_t i = 0; i < dof_size/value_size; i++)
		{
			data_flatten[i*value_size] = data[i].u1;
			data_flatten[i*value_size+1] = data[i].u2;
		}
		position.vector()->set_local(data_flatten);
	}

	void assign(std::vector<Particle<double>>&data, const Function &position){
		//TODO: check value_size == 2
		size_t value_size = position.value_size();        
		size_t dof_size = position.function_space()->dim();

		data.resize(dof_size/value_size);

		std::vector<double> data_flatten;
		position.vector()->get_local(data_flatten);

		// pack the data_flatten
		for (size_t i = 0; i < data.size(); i++)
		{
			data[i].u1 =  data_flatten[i*value_size];
			data[i].u2 =  data_flatten[i*value_size+1];
		}
	}

	void assign_coordinates(std::vector<Particle<double>>&data, const Function &position){
		//TODO: check value_size == 2
		size_t value_size = position.value_size();        
		size_t dof_size = position.function_space()->dim();

		data.resize(dof_size/value_size);

		std::vector<double> data_flatten;
		position.vector()->get_local(data_flatten);

		// pack the data_flatten
		for (size_t i = 0; i < data.size(); i++)
		{
			data[i].x =  data_flatten[i*value_size];
			data[i].y =  data_flatten[i*value_size+1];
		}
	}

	void fluid_to_solid(const Function &fluid, Function &solid)
	{

		std::vector<Particle<double>> array_solid;
		std::vector<double2> array_fluid;
		fluid_mesh->extract_dofs(array_fluid, fluid);
		fluid_mesh->interpolation(array_solid, array_fluid, current_coordinates);
		assign(solid, array_solid);
	}

	void solid_to_fluid(Function &fluid, const Function &solid)
	{
		std::vector<Particle<double>> array_solid;
		std::vector<double2> array_fluid;
		assign(array_solid, solid);
		for (size_t i = 0; i < array_solid.size(); i++)
		{
			array_solid[i].w = 1.0;
		}
		fluid_mesh->distribution(array_fluid, array_solid, current_coordinates);
		fluid_mesh->assign_dofs(array_fluid, fluid);
	}

};
