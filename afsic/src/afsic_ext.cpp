#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "mail/smtp_mail_sender.h"



namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(afsic_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
}


NB_MODULE(smtp_mail_sender, m) {
    nb::class_<smtp::EmailInfo>(m, "EmailInfo")
        .def(nb::init<>())
        .def_rw("smtp_url", &smtp::EmailInfo::smtp_url)
        .def_rw("username", &smtp::EmailInfo::username)
        .def_rw("password", &smtp::EmailInfo::password)
        .def_rw("from", &smtp::EmailInfo::from)
        .def_rw("to", &smtp::EmailInfo::to)
        .def_rw("subject", &smtp::EmailInfo::subject)
        .def_rw("body", &smtp::EmailInfo::body)
        .def_rw("is_html", &smtp::EmailInfo::is_html);

    m.def("send_email", &smtp::send_email, "Send an email");
}
