#include "coupling/main.h"
#include "mail/smtp_mail_sender.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;
using namespace nb::literals;

std::vector<double> fun(MPI_Comm comm, std::int32_t size_local, const double *data) {

    std::int32_t size_global(0);

    // start of gather
    int rank_root = 0;
    int mpi_rank;
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    MPI_Reduce(&size_local,                       // 发送缓冲区（每个进程的本地数据）
               &size_global,                      // 接收缓冲区（仅 root 有效）
               1,                                 // 数据数量
               dolfinx::MPI::mpi_t<std::int32_t>, // MPI 数据类型
               MPI_SUM,                           // 操作类型（求和）
               rank_root,                         // root 进程（rank=0）
               comm                               // MPI 通信子
    );

    std::vector<double> global_data(size_global);
    std::int32_t *size_local_s = nullptr;
    std::int32_t *displacement_s = nullptr;
    if (mpi_rank == rank_root) {
        size_local_s = (std::int32_t *)malloc(sizeof(std::int32_t) * mpi_size);
        displacement_s = (std::int32_t *)malloc(sizeof(std::int32_t) * mpi_size);
    }

    // 在收集变长数组之前，先计算偏移量
    MPI_Gather(&size_local, 1, dolfinx::MPI::mpi_t<std::int32_t>, size_local_s, 1, dolfinx::MPI::mpi_t<std::int32_t>, rank_root, comm);
    if (mpi_rank == rank_root) {
        displacement_s[0] = 0;
        for (int i = 1; i < mpi_size; i++) {
            displacement_s[i] = displacement_s[i - 1] + size_local_s[i - 1];
        }
        // total_data = displs[size-1] + recv_counts[size-1];
    }

    // Gather the data from all processes
    MPI_Gatherv(data, size_local, dolfinx::MPI::mpi_t<double>, global_data.data(), size_local_s, displacement_s, dolfinx::MPI::mpi_t<double>, rank_root, comm);

    if (mpi_rank == rank_root) {
        free(size_local_s);
        free(displacement_s);
    }
    return global_data;
}

std::int32_t reduce(MPI_Comm comm, std::int32_t size_local) {
    int rank_root = 0;
    int mpi_rank;
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    std::int32_t size_global(0);
    MPI_Reduce(&size_local,                       // 发送缓冲区（每个进程的本地数据）
               &size_global,                      // 接收缓冲区（仅 root 有效）
               1,                                 // 数据数量
               dolfinx::MPI::mpi_t<std::int32_t>, // MPI 数据类型
               MPI_SUM,                           // 操作类型（求和）
               rank_root,                         // root 进程（rank=0）
               comm                               // MPI 通信子
    );
    return size_global;
}

std::vector<double> scatter(MPI_Comm comm, std::int32_t size_local, const double *data) {

    // start of gather
    int rank_root = 0;
    int mpi_rank;
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    std::int32_t size_global(0);
    MPI_Reduce(&size_local,                       // 发送缓冲区（每个进程的本地数据）
               &size_global,                      // 接收缓冲区（仅 root 有效）
               1,                                 // 数据数量
               dolfinx::MPI::mpi_t<std::int32_t>, // MPI 数据类型
               MPI_SUM,                           // 操作类型（求和）
               rank_root,                         // root 进程（rank=0）
               comm                               // MPI 通信子
    );

    std::vector<double> local_data(size_local);
    std::int32_t *size_local_s = nullptr;
    std::int32_t *displacement_s = nullptr;
    if (mpi_rank == rank_root) {
        size_local_s = (std::int32_t *)malloc(sizeof(std::int32_t) * mpi_size);
        displacement_s = (std::int32_t *)malloc(sizeof(std::int32_t) * mpi_size);
    }

    // 在收集变长数组之前，先计算偏移量
    MPI_Gather(&size_local, 1, dolfinx::MPI::mpi_t<std::int32_t>, size_local_s, 1, dolfinx::MPI::mpi_t<std::int32_t>, rank_root, comm);
    if (mpi_rank == rank_root) {
        displacement_s[0] = 0;
        for (int i = 1; i < mpi_size; i++) {
            displacement_s[i] = displacement_s[i - 1] + size_local_s[i - 1];
        }
        // total_data = displs[size-1] + recv_counts[size-1];
    }

    // Gather the data from all processes
    MPI_Scatterv(data, size_local_s, displacement_s, dolfinx::MPI::mpi_t<double>, local_data.data(), size_local, dolfinx::MPI::mpi_t<double>, rank_root, comm);

    if (mpi_rank == rank_root) {
        free(size_local_s);
        free(displacement_s);
    }
    return local_data;
}

// int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
//     void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
//     int root, MPI_Comm comm)

// int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[],
//     MPI_Datatype sendtype, void *recvbuf, int recvcount,
//     MPI_Datatype recvtype, int root, MPI_Comm comm)

NB_MODULE(afsic_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);

    nb::class_<smtp::EmailInfo>(m, "EmailInfo")
        .def(nb::init<>())
        .def_rw("smtp_url", &smtp::EmailInfo::smtp_url)
        .def_rw("username", &smtp::EmailInfo::username)
        .def_rw("password", &smtp::EmailInfo::password)
        .def_rw("From", &smtp::EmailInfo::from)
        .def_rw("to", &smtp::EmailInfo::to)
        .def_rw("subject", &smtp::EmailInfo::subject)
        .def_rw("body", &smtp::EmailInfo::body)
        .def_rw("is_html", &smtp::EmailInfo::is_html);

    m.def("send_email", &smtp::send_email, "Send an email");

    m.def("coupling", &coupling::coupling, "IB coupling operators");

    nb::class_<coupling::IBMesh>(m, "IBMesh")
        .def(nb::init<double, double, double, double, std::int64_t, std::int64_t, uint>(), nb::arg("x0"), nb::arg("x1"), nb::arg("y0"), nb::arg("y1"),
             nb::arg("dim_x"), nb::arg("dim_y"), nb::arg("order"))
        .def(
            "build_map",
            [](coupling::IBMesh &self, nb::object py_coords) {
                nb::object x_attr = py_coords.attr("x");
                nb::object array_attr = x_attr.attr("array");
                auto data = nb::cast<nb::ndarray<T, nb::numpy>>(array_attr);

                // Print the array information if needed
                // printf("Array data pointer : %p\n", data.data());
                // printf("Array dimension : %zu\n", data.ndim());
                // for (size_t i = 0; i < data.ndim(); ++i) {
                //   printf("Array dimension [%zu] : %zu\n", i, data.shape(i));
                //   printf("Array stride    [%zu] : %zd\n", i, data.stride(i));
                // }
                // printf("Device ID = %u (cpu=%i, cuda=%i)\n", data.device_id(),
                //        int(data.device_type() == nb::device::cpu::value),
                //        int(data.device_type() == nb::device::cuda::value));
                // printf("Array dtype: int16=%i, uint32=%i, float32=%i, float64=%i\n",
                //        data.dtype() == nb::dtype<int16_t>(),
                //        data.dtype() == nb::dtype<uint32_t>(),
                //        data.dtype() == nb::dtype<float>(),
                //        data.dtype() == nb::dtype<double>());

                // Create a vector to store global data
                std::int32_t size_local = nb::cast<std::int32_t>(x_attr.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr.attr("bs"));

                auto global_data = fun(self.mesh()->comm(), size_local, data.data());

                int rank_root = 0;
                int mpi_rank;
                int mpi_size;
                MPI_Comm_size(self.mesh()->comm(), &mpi_size);
                MPI_Comm_rank(self.mesh()->comm(), &mpi_rank);

                if (mpi_rank == rank_root) {

                    // 最后在 root 进程调用 build_map 函数，IBMesh 中不用考虑 MPI 进程。
                    self.build_map(global_data);
                }
                printf("Global data size: %zu\n", global_data.size());
            },
            nb::arg("coords"), "build a map")
        .def(
            "evaluate",
            [](coupling::IBMesh &self, double x, double y, nb::object py_function) {
                nb::object x_attr = py_function.attr("x");
                nb::object array_attr = x_attr.attr("array");
                auto data = nb::cast<nb::ndarray<T, nb::numpy>>(array_attr);
                std::int32_t size_local = nb::cast<std::int32_t>(x_attr.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr.attr("bs"));
                auto function = fun(self.mesh()->comm(), size_local, data.data());

                int rank_root = 0;
                int mpi_rank;
                int mpi_size;
                MPI_Comm_size(self.mesh()->comm(), &mpi_size);
                MPI_Comm_rank(self.mesh()->comm(), &mpi_rank);
                if (mpi_rank == rank_root) {
                    auto result = self.evaluate(x, y, function);
                    for (const auto &val : result) {
                        printf("Result: %f\n", val);
                    }
                }
                // return nb::cast<std::vector<double>>(result);
            },
            "evaluate a function at a point", nb::arg("x"), nb::arg("y"), nb::arg("function"));

    nb::class_<coupling::IBInterpolation>(m, "IBInterpolation")
        .def(nb::init<coupling::IBMesh &>(), nb::arg("ibmesh"))
        .def(
            "evaluate_current_points",
            [](coupling::IBInterpolation &self, nb::object py_coords) {
                nb::object x_attr = py_coords.attr("x");
                nb::object array_attr = x_attr.attr("array");
                auto data = nb::cast<nb::ndarray<T, nb::numpy>>(array_attr);
                std::int32_t size_local = nb::cast<std::int32_t>(x_attr.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr.attr("bs"));
                const auto global_data = fun(self.fluid_mesh.mesh()->comm(), size_local, data.data());

                int rank_root = 0;
                int mpi_rank;
                int mpi_size;
                MPI_Comm_size(self.fluid_mesh.mesh()->comm(), &mpi_size);
                MPI_Comm_rank(self.fluid_mesh.mesh()->comm(), &mpi_rank);

                if (mpi_rank == rank_root) {
                    self.evaluate_current_points(global_data);
                }
                printf("Global data size: %zu\n", global_data.size());
            },
            nb::arg("position"))
        .def(
            "fluid_to_solid",
            [](coupling::IBInterpolation &self, nb::object py_fluid, nb::object py_solid) {
                auto comm = self.fluid_mesh.mesh()->comm();
                int rank_root = 0;
                int mpi_rank;
                int mpi_size;
                MPI_Comm_size(comm, &mpi_size);
                MPI_Comm_rank(comm, &mpi_rank);

                // Collect fluid data
                nb::object x_attr_f = py_fluid.attr("x");
                nb::object array_attr_f = x_attr_f.attr("array");
                auto data_f = nb::cast<nb::ndarray<T, nb::numpy>>(array_attr_f);
                std::int32_t size_local_f = nb::cast<std::int32_t>(x_attr_f.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr_f.attr("bs"));
                std::vector<double> global_data_f = fun(comm, size_local_f, data_f.data());

                // Collect solid data
                nb::object x_attr_s = py_solid.attr("x");
                std::int32_t size_local_s = nb::cast<std::int32_t>(x_attr_s.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr_s.attr("bs"));

                // Fluid to solid
                auto size_global_s = reduce(comm, size_local_s);
                std::vector<double> global_data_s(size_global_s);
                if (mpi_rank == rank_root) {
                    self.fluid_to_solid(global_data_f, global_data_s);
                }

                // Scatter the solid data back to all processes
                auto local_data_s = scatter(comm, size_local_s, global_data_s.data());

                // Assign the solid data to the py_solid object
                auto data_s = nb::cast<nb::ndarray<T, nb::numpy>>(x_attr_s.attr("array"));
                std::memcpy(data_s.data(), local_data_s.data(), local_data_s.size() * sizeof(double));
            },
            nb::arg("fluid"), nb::arg("solid"))
        .def(
            "solid_to_fluid",
            [](coupling::IBInterpolation &self, nb::object py_fluid, nb::object py_solid) {
                auto comm = self.fluid_mesh.mesh()->comm();
                int rank_root = 0;
                int mpi_rank;
                int mpi_size;
                MPI_Comm_size(comm, &mpi_size);
                MPI_Comm_rank(comm, &mpi_rank);

                // Collect fluid data
                nb::object x_attr_f = py_fluid.attr("x");
                std::int32_t size_local_f = nb::cast<std::int32_t>(x_attr_f.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr_f.attr("bs"));

                // Collect solid data
                nb::object x_attr_s = py_solid.attr("x");
                std::int32_t size_local_s = nb::cast<std::int32_t>(x_attr_s.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr_s.attr("bs"));
                nb::object array_attr_s = x_attr_s.attr("array");
                auto data_s = nb::cast<nb::ndarray<T, nb::numpy>>(array_attr_s);
                std::vector<double> global_data_s = fun(comm, size_local_s, data_s.data());

                // Solid to fluid
                auto size_global_f = reduce(comm, size_local_f);
                std::vector<double> global_data_f(size_global_f);
                if (mpi_rank == rank_root) {
                    self.solid_to_fluid(global_data_f, global_data_s);
                }

                // Scatter the fluid data back to all processes
                auto local_data_f = scatter(comm, size_local_f, global_data_f.data());

                // Assign the solid data to the py_solid object
                auto data_f = nb::cast<nb::ndarray<T, nb::numpy>>(x_attr_f.attr("array"));
                std::memcpy(data_f.data(), local_data_f.data(), local_data_f.size() * sizeof(double));
            },
            nb::arg("fluid"), nb::arg("solid"));
}
