/// @date 2025-01-05
/// @file FacetIntegration.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2025 Ma Pengfei
///
/// @brief
///
///
#include <dolfin.h>

namespace dolfin
{

    class FacetIntegration
    {
    private:
        std::vector<std::size_t> _facets;
        const size_t num_points = 1;
        const std::vector<std::vector<double>> some_points_all = {{0.5, 0.5}, {0.0, 0.5}, {0.5, 0.0}};

    public:
        // Find all id of facets on the given sub_domain
        FacetIntegration(const std::shared_ptr<Mesh> &mesh, const std::shared_ptr<MeshFunction<std::size_t>> &bdry, size_t sub_domain);

        std::pair<std::vector<double>, std::vector<double>> fun3(const std::shared_ptr<Function> &disp, const std::shared_ptr<Function> &force);

        void fun4(const std::shared_ptr<Function> &disp, const std::shared_ptr<Function> &force);

        ~FacetIntegration()
        {
        }
    };
} // namespace dolfin
