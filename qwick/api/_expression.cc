#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "index.h"
#include "operator.h"
#include "expression.h"

namespace py = pybind11;

namespace pybind11::literals {

void export_expression(py::module &m) {
    // FIXME: not exposing the data keyword since it's currently not a set
    py::class_<TermMap, std::shared_ptr<TermMap>>(m, "TermMap")
        .def(py::init<std::vector<Sigma>, std::vector<Tensor>>())
        .def("__eq__", [](TermMap &a, TermMap &b) { return a == b; })
        .def("__neq__", [](TermMap &a, TermMap &b) { return a != b; });

    py::class_<Term, std::shared_ptr<Term>>(m, "Term")
        .def(py::init([](
                double scalar,
                std::vector<Sigma> sums,
                std::vector<Tensor> tensors,
                std::vector<Operator> operators,
                std::vector<Delta> deltas,
                std::unordered_map<std::string, std::string> index_key) {
                    return Term(scalar, sums, tensors, operators, deltas, index_key);
                }),
                py::arg("scalar")=1.0,
                py::arg("sums")=std::vector<Sigma>(),
                py::arg("tensors")=std::vector<Tensor>(),
                py::arg("operators")=std::vector<Operator>(),
                py::arg("deltas")=std::vector<Delta>(),
                py::arg("index_key")=default_index_key
        )
        .def(py::init([](
                double scalar,
                std::vector<Sigma> sums,
                std::vector<Tensor> tensors,
                std::vector<Operator> operators,
                std::vector<Delta> deltas,
                py::none index_key) {
                    return Term(scalar, sums, tensors, operators, deltas);
                }),
                py::arg("scalar")=1.0,
                py::arg("sums")=std::vector<Sigma>(),
                py::arg("tensors")=std::vector<Tensor>(),
                py::arg("operators")=std::vector<Operator>(),
                py::arg("deltas")=std::vector<Delta>(),
                py::arg("index_key")=py::none()
        )
        .def_readwrite("scalar", &Term::scalar)
        .def_readwrite("sums", &Term::sums)
        .def_readwrite("tensors", &Term::tensors)
        .def_readwrite("operators", &Term::operators)
        .def_readwrite("deltas", &Term::deltas)
        .def_readwrite("index_key", &Term::index_key)
        .def("__repr__", &Term::repr)
        .def("__str__", &Term::repr)
        .def("__eq__", [](const Term &a, const Term &b) { return a == b; })
        .def("__ne__", [](const Term &a, const Term &b) { return a != b; })
        .def("__mul__", [](Term &a, Term &b) { return a * b; })
        .def("__mul__", [](Term &a, double &b) { return a * b; })
        .def("__rmul__", [](Term &a, double &b) { return a * b; })
        .def("__mul__", [](Term &a, int &b) { return a * b; })
        .def("__rmul__", [](Term &a, int &b) { return a * b; })
        .def("_print_str", &Term::_print_str, py::arg("scalar")=true)
        .def("_idx_map", &Term::_idx_map)
        .def("ilist", &Term::ilist)
        .def("resolve", &Term::resolve)
        .def("_inc", &Term::_inc)
        .def("copy", &Term::copy);

    py::class_<ATerm, std::shared_ptr<ATerm>>(m, "ATerm")
        .def(py::init<Term>())
        .def(py::init([](
                double scalar,
                std::vector<Sigma> sums,
                std::vector<Tensor> tensors,
                std::unordered_map<std::string, std::string> index_key) {
                    return ATerm(scalar, sums, tensors, index_key);
                }),
                py::arg("scalar")=1.0,
                py::arg("sums")=std::vector<Sigma>(),
                py::arg("tensors")=std::vector<Tensor>(),
                py::arg("index_key")=default_index_key
        )
        .def(py::init([](
                double scalar,
                std::vector<Sigma> sums,
                std::vector<Tensor> tensors,
                py::none index_key) {
                    return ATerm(scalar, sums, tensors);
                }),
                py::arg("scalar")=1.0,
                py::arg("sums")=std::vector<Sigma>(),
                py::arg("tensors")=std::vector<Tensor>(),
                py::arg("index_key")=py::none()
        )
        .def_readwrite("scalar", &ATerm::scalar)
        .def_readwrite("sums", &ATerm::sums)
        .def_readwrite("tensors", &ATerm::tensors)
        .def_readwrite("index_key", &ATerm::index_key)
        .def("__repr__", &ATerm::repr)
        .def("__str__", &ATerm::repr)
        .def("__eq__", [](const ATerm &a, const ATerm &b) { return a == b; })
        .def("__ne__", [](const ATerm &a, const ATerm &b) { return a != b; })
        .def("__lt__", [](const ATerm &a, const ATerm &b) { return a < b; })
        .def("__le__", [](const ATerm &a, const ATerm &b) { return a <= b; })
        .def("__gt__", [](const ATerm &a, const ATerm &b) { return a > b; })
        .def("__ge__", [](const ATerm &a, const ATerm &b) { return a >= b; })
        .def("__mul__", [](ATerm &a, ATerm &b) { return a * b; })
        .def("__mul__", [](ATerm &a, double &b) { return a * b; })
        .def("__rmul__", [](ATerm &a, double &b) { return a * b; })
        .def("__mul__", [](ATerm &a, int &b) { return a * b; })
        .def("__rmul__", [](ATerm &a, int &b) { return a * b; })
        .def("_print_str", &ATerm::_print_str, py::arg("scalar")=true)
        .def("_idx_map", &ATerm::_idx_map)
        .def("_einsum_str", &ATerm::_einsum_str)
        .def("_inc", &ATerm::_inc)
        .def("match", &ATerm::match)
        .def("pmatch", &ATerm::pmatch)
        .def("ilist", &ATerm::ilist)
        .def("nidx", &ATerm::nidx)
        .def("sort_tensors", &ATerm::sort_tensors)
        .def("merge_external", &ATerm::merge_external)
        .def("connected", &ATerm::connected)
        .def("reducible", &ATerm::reducible)
        .def("transpose", &ATerm::transpose)
        .def("copy", &ATerm::copy);

    py::class_<Expression, std::shared_ptr<Expression>>(m, "Expression")
        .def(py::init<std::vector<Term>>())
        .def_readwrite("terms", &Expression::terms)
        .def_readonly("tthresh", &Expression::tthresh)
        .def("__repr__", &Expression::repr)
        .def("__str__", &Expression::repr)
        .def("__eq__", [](const Expression &a, const Expression &b) { return a == b; })
        .def("__ne__", [](const Expression &a, const Expression &b) { return a != b; })
        .def("__add__", [](const Expression &a, const Expression &b) { return a + b; })
        .def("__sub__", [](Expression &a, Expression &b) { return a - b; })
        .def("__mul__", [](Expression &a, Expression &b) { return a * b; })
        .def("__mul__", [](Expression &a, double &b) { return a * b; })
        .def("__rmul__", [](Expression &a, double &b) { return a * b; })
        .def("__mul__", [](Expression &a, int &b) { return a * b; })
        .def("__rmul__", [](Expression &a, int &b) { return a * b; })
        .def("_print_str", &Expression::_print_str)
        .def("resolve", &Expression::resolve)
        .def("repr", &Expression::repr)
        .def("are_operators", &Expression::are_operators);

    py::class_<AExpression, std::shared_ptr<AExpression>>(m, "AExpression")
        .def(py::init<std::vector<ATerm>, bool, bool>())
        .def(py::init([](
                Expression Ex,
                bool simplify,
                bool sort) {
                    return AExpression(Ex, simplify, sort);
                }),
                py::arg("Ex"),
                py::arg("simplify")=true,
                py::arg("sort")=true
        )
        .def(py::init([](
                std::vector<ATerm> terms,
                bool simplify,
                bool sort) {
                    return AExpression(terms, simplify, sort);
                }),
                py::arg("terms")=std::vector<Term>(),
                py::arg("simplify")=true,
                py::arg("sort")=true
        )
        .def_readwrite("terms", &AExpression::terms)
        .def_readonly("tthresh", &AExpression::tthresh)
        .def("__repr__", &AExpression::_print_str)
        .def("simplify", &AExpression::simplify)
        .def("sort", &AExpression::sort)
        .def("sort_tensors", &AExpression::sort_tensors)
        .def("_print_str", &AExpression::_print_str)
        .def("_print_einsum", &AExpression::_print_einsum, py::arg("lhs")="")
        .def("connected", &AExpression::connected)
        .def("get_connected", &AExpression::get_connected,
                py::arg("simplify")=true, py::arg("sort")=true)
        .def("pmatch", &AExpression::pmatch)
        .def("transpose", &AExpression::transpose);

    m.def("get_case", &get_case, "dd"_a, "ilist"_a);
    m.def("_resolve", &_resolve, "sums"_a, "tensors"_a, "operators"_a, "deltas"_a);
}

PYBIND11_MODULE(_expression, m) {
    m.attr("__name__") = "qwick._expression";
    m.doc() = "C++ interface to qwick._expression";

    export_expression(m);
}

}  // namespace pybind11:literals
