#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "index.h"
#include "operator.h"
#include "expression.h"
#include "wick.h"

namespace py = pybind11;

namespace pybind11::literals {

void export_wick(py::module &m) {
    m.def("valid_contraction", &valid_contraction, "o1"_a, "o2"_a);
    m.def("pair_list", &pair_list, "lst"_a);
    m.def("find_pair", &find_pair, "i"_a, "ipairs"_a);
    m.def("get_sign", &get_sign, "ipairs"_a);
    m.def("split_operators", &split_operators, "ops"_a);
    m.def("apply_wick", &apply_wick, "e"_a);
}

PYBIND11_MODULE(_wick, m) {
    m.attr("__name__") = "cwick._wick";
    m.doc() = "C++ interface to cwick._wick";

    export_wick(m);
}

}  // namespace pybind11::literals
