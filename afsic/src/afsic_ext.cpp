#include "coupling/main.h"
#include "mail/smtp_mail_sender.h"

#include <nanobind/nanobind.h>
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
      .def(nb::init<double, double, double, double,std::int64_t, std::int64_t, uint>(),
           nb::arg("x0"), nb::arg("x1"), nb::arg("y0"), nb::arg("y1"),
           nb::arg("dim_x"), nb::arg("dim_y"), nb::arg("order"));
}
