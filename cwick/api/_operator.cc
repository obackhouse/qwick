#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "index.h"
#include "operator.h"

namespace py = pybind11;

namespace pybind11::literals {

void export_operator(py::module &m) {
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator")
        .def(py::init([]() { return Operator(); }))
        .def(py::init([](Idx _idx, bool _ca, bool _fermion) { return Operator(_idx, _ca, _fermion); }))
        .def_readwrite("idx", &Operator::idx)
        .def_readwrite("ca", &Operator::ca)
        .def_readonly("projector", &Operator::projector)
        .def_readonly("fermion", &Operator::fermion)
        .def("__repr__", &Operator::repr)
        .def("__str__", &Operator::repr)
        .def("__eq__", [](const Operator &a, const Operator &b) { return a == b; })
        .def("__ne__", [](const Operator &a, const Operator &b) { return a != b; })
        .def("_print_str", &Operator::_print_str)
        .def("_inc", &Operator::_inc)
        .def("dagger", &Operator::dagger)
        .def("copy", &Operator::copy)
        .def("qp_creation", &Operator::qp_creation)
        .def("qp_annihilation", &Operator::qp_annihilation);

    py::class_<TensorSym, std::shared_ptr<TensorSym>>(m, "TensorSym")
        .def(py::init<std::vector<std::vector<int>>, std::vector<int>>())
        .def_readwrite("plist", &TensorSym::plist)
        .def_readwrite("signs", &TensorSym::signs)
        .def_readwrite("tlist", &TensorSym::tlist);

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<const std::vector<Idx>, const std::string, const TensorSym>(),
                py::arg("indices"), py::arg("name"), py::arg("sym")=TensorSym())
        .def_readwrite("indices", &Tensor::indices)
        .def_readwrite("name", &Tensor::name)
        .def_readwrite("sym", &Tensor::sym)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("__eq__", [](const Tensor &a, const Tensor &b) { return a == b; })
        .def("__ne__", [](const Tensor &a, const Tensor &b) { return a != b; })
        .def("__lt__", [](const Tensor &a, const Tensor &b) { return a < b; })
        .def("__le__", [](const Tensor &a, const Tensor &b) { return a <= b; })
        .def("__gt__", [](const Tensor &a, const Tensor &b) { return a > b; })
        .def("__ge__", [](const Tensor &a, const Tensor &b) { return a >= b; })
        .def("_inc", &Tensor::_inc)
        .def("_istr", &Tensor::_istr)
        .def("_print_str", &Tensor::_print_str)
        .def("transpose", &Tensor::transpose)
        .def("copy", &Tensor::copy);

    py::class_<Sigma, std::shared_ptr<Sigma>>(m, "Sigma")
        .def(py::init<Idx>())
        .def_readwrite("idx", &Sigma::idx)
        .def("__repr__", &Sigma::repr)
        .def("__str__", &Sigma::repr)
        .def("__eq__", [](const Sigma &a, const Sigma &b) { return a == b; })
        .def("__ne__", [](const Sigma &a, const Sigma &b) { return a != b; })
        .def("__lt__", [](const Sigma &a, const Sigma &b) { return a < b; })
        .def("__le__", [](const Sigma &a, const Sigma &b) { return a <= b; })
        .def("__gt__", [](const Sigma &a, const Sigma &b) { return a > b; })
        .def("__ge__", [](const Sigma &a, const Sigma &b) { return a >= b; })
        .def("_inc", &Sigma::_inc)
        .def("_print_str", &Sigma::_print_str)
        .def("copy", &Sigma::copy);

    py::class_<Delta, std::shared_ptr<Delta>>(m, "Delta")
        .def(py::init<Idx, Idx>())
        .def_readwrite("i1", &Delta::i1)
        .def_readwrite("i2", &Delta::i2)
        .def("__eq__", [](const Delta &a, const Delta &b) { return a == b; })
        .def("__ne__", [](const Delta &a, const Delta &b) { return a != b; })
        .def("__repr__", &Delta::repr)
        .def("__str__", &Delta::repr)
        .def("_inc", &Delta::_inc)
        .def("_print_str", &Delta::_print_str)
        .def("copy", &Delta::copy);
    
    m.def("permute", &permute, "t"_a, "p"_a);
    m.def("tensor_from_delta", &tensor_from_delta, "d"_a);
    m.def("is_normal_ordered", &is_normal_ordered, "operators"_a);
    m.def("normal_ordered", &normal_ordered, "operators"_a, "sign"_a=1);
}

PYBIND11_MODULE(_operator, m) {
    m.attr("__name__") = "cwick._operator";
    m.doc() = "C++ interface to cwick._operator";

    export_operator(m);
}

}  // namespace pybind11::literals
