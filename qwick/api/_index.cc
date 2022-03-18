#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "index.h"

namespace py = pybind11;

void export_index(py::module &m) {
    py::class_<Idx, std::shared_ptr<Idx>>(m, "Idx")
        .def(py::init<int, std::string, bool>(), py::arg("index"), py::arg("space"), py::arg("fermion")=true)
        .def_readwrite("index", &Idx::index)
        .def_readwrite("space", &Idx::space)
        .def_readwrite("fermion", &Idx::fermion)
        .def("__repr__", &Idx::repr)
        .def("__str__", &Idx::repr)
        .def("__hash__", [](const Idx _idx) { return IdxHash()(_idx); })
        .def("__eq__", [](const Idx &a, const Idx &b) { return a == b; })
        .def("__ne__", [](const Idx &a, const Idx &b) { return a != b; })
        .def("__lt__", [](const Idx &a, const Idx &b) { return a < b; })
        .def("__le__", [](const Idx &a, const Idx &b) { return a <= b; })
        .def("__gt__", [](const Idx &a, const Idx &b) { return a > b; })
        .def("__ge__", [](const Idx &a, const Idx &b) { return a >= b; });

    m.def("idx_copy", &idx_copy, py::arg("a"));
    m.def("is_occupied", &is_occupied, py::arg("a"), py::arg("occ")=py::list());
}

PYBIND11_MODULE(_index, m) {
    m.attr("__name__") = "qwick._index";
    m.doc() = "C++ interface to qwick._index";

    export_index(m);
}
