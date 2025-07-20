#include "coupling/main.h"
#include "mail/smtp_mail_sender.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;
using namespace nb::literals;

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
      .def(nb::init<double, double, double, double, std::int64_t, std::int64_t,
                    uint>(),
           nb::arg("x0"), nb::arg("x1"), nb::arg("y0"), nb::arg("y1"),
           nb::arg("dim_x"), nb::arg("dim_y"), nb::arg("order"))
      .def(
          "build_map",
          [](coupling::IBMesh &self, nb::object py_coords) {
            nb::object x_attr = py_coords.attr("x");
            nb::object array_attr = x_attr.attr("array");
            auto data = nb::cast<nb::ndarray<T, nb::numpy>>(array_attr);

            printf("Array data pointer : %p\n", data.data());
            printf("Array dimension : %zu\n", data.ndim());
            for (size_t i = 0; i < data.ndim(); ++i) {
              printf("Array dimension [%zu] : %zu\n", i, data.shape(i));
              printf("Array stride    [%zu] : %zd\n", i, data.stride(i));
            }
            printf("Device ID = %u (cpu=%i, cuda=%i)\n", data.device_id(),
                   int(data.device_type() == nb::device::cpu::value),
                   int(data.device_type() == nb::device::cuda::value));
            printf("Array dtype: int16=%i, uint32=%i, float32=%i, float64=%i\n",
                   data.dtype() == nb::dtype<int16_t>(),
                   data.dtype() == nb::dtype<uint32_t>(),
                   data.dtype() == nb::dtype<float>(),
                   data.dtype() == nb::dtype<double>());

            // std::cout << "MPI Comm: " << MPI_COMM_WORLD << std::endl;
            std::cout << "MPI Comm: " << self.mesh()->comm() << std::endl;

            // create a vector to put global data
            // NOTE: 我不知道用 data.shape(0); 表示数组总长是否合适。
            std::int32_t size_local = data.shape(0);
            std::int32_t size_global(0);
            MPI_Allreduce(&size_local, &size_global, 1,
                          dolfinx::MPI::mpi_t<std::int32_t>, MPI_SUM,
                          self.mesh()->comm());
            std::vector<double> global_data(size_global);

            // 看看对了没
            int mpi_rank;
            MPI_Comm_rank(self.mesh()->comm(), &mpi_rank);
            printf("%d %d %d", mpi_rank, size_local, size_global);
          },
          nb::arg("coords"), "build a map");
  // ,
  //     .def(
  //   "interpolate",
  //   [](dolfinx::fem::Function<T, U>& self,
  //      dolfinx::fem::Function<T, U>& u0,
  //      nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells0,
  //      nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells1)
  //   {
  //     self.interpolate(u0, std::span(cells0.data(), cells0.size()),
  //                      std::span(cells1.data(), cells1.size()));
  //   },
  //   nb::arg("u"), nb::arg("cells0"), nb::arg("cells1"),
  //   "Interpolate a finite element function")
}
