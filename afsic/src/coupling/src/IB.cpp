/// @date 2024-07-12
/// @file IB.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief 封装IBInterpolation和IBMesh类，提供python接口
/// 
///
#include "IBMesh3D.h"
#include "IBInterpolation3D.h"
#include "gauss_quadrature/FacetIntegration.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


using namespace dolfin;

namespace py = pybind11;
PYBIND11_MODULE(cpp, m)
{
	py::class_<double2>(m, "double2")
		.def_readonly("x", &double2::x)
		.def_readonly("y", &double2::y);

    py::class_<IBMesh, std::shared_ptr<IBMesh>>(m, "IBMesh")
        .def(py::init<std::array<dolfin::Point, 2>, std::array<size_t, 2>, size_t>())
		.def("mesh",&IBMesh::mesh)
		.def("get_hash",&IBMesh::get_hash)
 		.def("build_map", &IBMesh::build_map)
		.def("evaluate", &IBMesh::evaluate)
		;

	py::class_<IBMesh3D, std::shared_ptr<IBMesh3D>>(m, "IBMesh3D")
        .def(py::init<std::array<dolfin::Point, 2>, std::array<size_t, 3>, size_t>())
		.def("mesh",&IBMesh3D::mesh)
		.def("get_hash",&IBMesh3D::get_hash)
		.def("build_map", &IBMesh3D::build_map)
		.def("evaluate", &IBMesh3D::evaluate)
		;

    py::class_<FacetIntegration>(m, "FacetIntegration")
        .def(py::init<std::shared_ptr<Mesh>, std::shared_ptr<MeshFunction<std::size_t>>, size_t>())
		.def("fun4", &FacetIntegration::fun4)
		.def("fun3", &FacetIntegration::fun3)
		;

    py::class_<IBInterpolation>(m, "IBInterpolation")
        .def(py::init<std::shared_ptr<IBMesh>, std::shared_ptr<Mesh>>())
		.def("solid_to_fluid", &IBInterpolation::solid_to_fluid)
		.def("fluid_to_solid", &IBInterpolation::fluid_to_solid)
		.def("evaluate_current_points", &IBInterpolation::evaluate_current_points)
		;
	
	py::class_<IBInterpolation3D>(m, "IBInterpolation3D")
        .def(py::init<std::shared_ptr<IBMesh3D>, std::shared_ptr<Mesh>>())
		.def("solid_to_fluid", &IBInterpolation3D::solid_to_fluid)
		.def("fluid_to_solid", &IBInterpolation3D::fluid_to_solid)
		.def("evaluate_current_points", &IBInterpolation3D::evaluate_current_points)
		;


	
	#ifdef VERSION_INFO
		m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
	#else
		m.attr("__version__") = "dev";
	#endif
}