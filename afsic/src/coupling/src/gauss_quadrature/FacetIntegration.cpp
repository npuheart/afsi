/// @date 2025-01-05
/// @file FacetIntegration.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2025 Ma Pengfei
///
/// @brief
///
///
#define DEBUG
#include "VTUIO.h"
#include "FacetIntegration.h"
using namespace dolfin;



// 排列方式：
// p1_x, p1_y, p2_x, p2_y, p3_x, p3_y

template <size_t DIM>
void linearInterpolation(double *points_local, const double *points_ref, const double *vertices, size_t num_points)
{
    for (std::size_t i = 0; i < num_points; ++i)
    {
        double alpha = 1.0 - points_ref[DIM * i] - points_ref[DIM * i + 1];
        double beta = points_ref[DIM * i];
        double gamma = points_ref[DIM * i + 1];

        for (size_t j = 0; j < DIM; j++)
        {

            points_local[DIM * i + j] = alpha * vertices[j] + beta * vertices[DIM + j] + gamma * vertices[2 * DIM + j];
            // printf("alpha: %f, beta: %f, gamma: %f\n", alpha, beta, gamma);
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> FacetIntegration::fun3(const std::shared_ptr<Function> &disp, const std::shared_ptr<Function> &force)
{
    // TODO: check if disp and force are on the same function space
    auto _function_space = disp->function_space();
    const Mesh &mesh = *_function_space->mesh();
    const std::size_t D = mesh.topology().dim();
    const GenericDofMap &dofmap = *_function_space->dofmap();
    const FiniteElement &element = *_function_space->element();
    const ufc::finite_element &ufc_element = *element.ufc_element();

    dolfin_assert(D==2 );


    ufc::cell ufc_cell;
    auto value_size = disp->value_size();
    auto space_dimension = element.space_dimension();
    std::vector<double> basis_values(value_size * space_dimension);
    std::vector<double> coefficients_disp(element.space_dimension());
    std::vector<double> coefficients_force(element.space_dimension());

    // to hold the results
    std::vector<double> values_disp_all(_facets.size() * num_points * value_size);
    std::vector<double> values_force_all(_facets.size() * num_points * value_size);

    for (std::size_t f = 0; f < _facets.size(); ++f)
    {
        // Create facet
        const Facet facet(mesh, _facets[f]);

        // Get cell to which facet belongs.
        dolfin_assert(facet.num_entities(D) > 0);
        const std::size_t cell_index = facet.entities(D)[0];
        const Cell cell(mesh, cell_index);

        // Get local index of facet with respect to the cell
        const std::size_t local_facet = cell.index(facet);
        // TODO: Get Gauss Quadrature Rule for local facet.
        const auto some_points = some_points_all[local_facet];
        // printf("local_facet: %ld\n", local_facet);

        auto dofs = dofmap.cell_dofs(cell_index);
        disp->vector()->get_local(coefficients_disp.data(), dofs.size(), dofs.data());
        force->vector()->get_local(coefficients_force.data(), dofs.size(), dofs.data());

        std::vector<double> vertex_coordinates(6);
        cell.get_vertex_coordinates(vertex_coordinates);
        // printf("vertex_coordinates: %f %f, %f %f, %f %f\n",
        //        vertex_coordinates[0], vertex_coordinates[1], vertex_coordinates[2],
        //        vertex_coordinates[3], vertex_coordinates[4], vertex_coordinates[5]);

        std::vector<double> some_points_local(some_points.size());
        linearInterpolation<2>(some_points_local.data(), some_points.data(), vertex_coordinates.data(), num_points);
        std::vector<double> ref_vertex_basis_values(value_size * space_dimension * num_points);
        ufc_element.evaluate_reference_basis(ref_vertex_basis_values.data(), num_points, some_points.data());

        std::vector<double> values_disp(value_size * num_points);
        std::vector<double> values_force(value_size * num_points);

        for (size_t k = 0; k < num_points; k++)
        {
            for (std::size_t i = 0; i < space_dimension; ++i)
            {
                for (std::size_t j = 0; j < value_size; ++j)
                {
                    values_disp[value_size * k + j] += coefficients_disp[i] * ref_vertex_basis_values[value_size * space_dimension * k + value_size * i + j];
                    values_force[value_size * k + j] += coefficients_force[i] * ref_vertex_basis_values[value_size * space_dimension * k + value_size * i + j];
                }
            }
        }

        for (size_t k = 0; k < num_points; k++)
        {
            for (std::size_t j = 0; j < value_size; ++j)
            {
                values_disp_all[f * num_points * value_size + value_size * k + j] = values_disp[value_size * k + j];
                values_force_all[f * num_points * value_size + value_size * k + j] = values_force[value_size * k + j];
            }
        }
        // for (size_t i = 0; i < values.size(); i++)
        // {
        //     printf("%f ", values[i]);
        // }
        // printf("\n");

        // for (size_t i = 0; i < 2 * num_points; i++)
        // {
        //     printf("%.e ", some_points_local[i]);
        // }
        // printf("\n");
        // for (size_t i = 0; i < 2 * num_points; i++)
        // {
        //     printf("%.e ", values_disp[i]);
        // }
        // printf("\n");
        // for (size_t i = 0; i < value_size * num_points; i++)
        // {
        //     printf("%.e ", values_force[i] - values_disp[i] * values_disp[i]);
        // }
        // printf("\n");
        // for (size_t i = 0; i < 2 * num_points; i++)
        // {
        //     printf("%f ", some_points[i]);
        // }
        // printf("\n");
    }
    return std::make_pair(values_disp_all, values_force_all);
}

void FacetIntegration::fun4(const std::shared_ptr<Function> &disp, const std::shared_ptr<Function> &force)
{
    auto ret = fun3(disp, force);
    std::vector<std::array<double, 3>> points0;
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<double, 3>> velocities;

    size_t num_points = ret.first.size() / disp->value_size();
    for (size_t i = 0; i < num_points; i++)
    {
        points0.push_back({ret.first[2 * i], ret.first[2 * i + 1], 0.0});
        points.push_back({ret.first[2 * i], ret.first[2 * i + 1], 0.0});
        velocities.push_back({ret.first[2 * i], ret.first[2 * i + 1], 0.0});
    }
    std::vector<std::array<double, 3>> data1 = points;
    std::vector<std::array<double, 3>> data2 = velocities;

    std::vector<std::string> tags{"position"};
    IO::write_particles_to_vtu("a.vtu", points, tags, data2);
}

// Find all id of facets on the given sub_domain
FacetIntegration::FacetIntegration(const std::shared_ptr<Mesh> &mesh, const std::shared_ptr<MeshFunction<std::size_t>> &bdry, size_t sub_domain)
{
    // Make sure we have the facet - cell connectivity
    const std::size_t D = mesh->topology().dim();
    mesh->init(D - 1); // Is it neccessary ?
    mesh->init(D);
    mesh->init(D - 1, D);

    // Build set of boundary facets
    // TODO: find facets on the boundary
    dolfin_assert(_facets.empty());
    for (FacetIterator facet(*mesh); !facet.end(); ++facet)
    {
        if ((*bdry)[*facet] == sub_domain)
            _facets.push_back(facet->index());
    }
}