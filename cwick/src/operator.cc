/*
 *  Operators
 */

#include "index.h"
#include "operator.h"


Operator::Operator() {
    // Projector
    projector = true;
}

Operator::Operator(const Idx _idx, bool _ca, bool _fermion) {
    // FOperator or BOperator
    idx = _idx;
    ca = _ca;
    projector = false;
    fermion = _fermion;
}

std::string Operator::repr() {
    std::string out;

    if (projector) {
        out = "P";
    }
    else if (fermion) {
        if (ca) {
            out = "a^{\\dagger}_" + idx.repr();
        }
        else {
            out = "a_" + idx.repr();
        }
    }
    else {
        if (ca) {
            out = "b^{\\dagger}_" + idx.repr();
        }
        else {
            out = "b_" + idx.repr();
        }
    }

    return out;
}

Operator Operator::_inc(int i) {
    if (projector) {
        return *this;
    }
    else {
        Idx new_idx(idx.index + i, idx.space, fermion);
        return Operator(new_idx, ca, fermion);
    }
}

Operator Operator::dagger() {
    if (projector) {
        return *this;
    }
    else {
        return Operator(idx_copy(idx), !(ca), fermion);
    }
}

Operator Operator::copy() {
    if (projector) {
        return *this;
    }
    else {
        return Operator(idx_copy(idx), ca, fermion);
    }
}

bool operator==(const Operator &a, const Operator &b) {
    if (a.projector || b.projector) {
        return (a.projector && b.projector);
    }
    else {
        return ((a.fermion == b.fermion) && (a.idx == b.idx) && (a.ca == b.ca));
    }
}

bool operator!=(const Operator &a, const Operator &b) {
    return (!(a == b));
}

bool Operator::qp_creation() {
    bool out;

    assert(!(projector));

    if (fermion) {
        if (!(is_occupied(idx)) && ca) {
            out = true;
        }
        else if (is_occupied(idx) and !(ca)) {
            out = true;
        }
        else {
            out = false;
        }
    }
    else {
        out = ca;
    }

    return out;
}

bool Operator::qp_annihilation() {
    return (!(qp_creation()));
}


// TensorSym

TensorSym::TensorSym() {}

TensorSym::TensorSym(
        const std::vector<std::vector<int>> _plist,
        std::vector<int> _signs)
{
    assert(_plist.size() == _signs.size());

    plist = _plist;
    signs = _signs;

    for (unsigned int i = 0; i < plist.size(); i++) {
        std::pair<std::vector<int>, int> p(plist[i], signs[i]);
        tlist.push_back(p);
    }
}


// Tensor

Tensor::Tensor() {};

Tensor::Tensor(
        const std::vector<Idx> _indices,
        const std::string _name,
        const TensorSym _sym)
{
    indices = _indices;
    name = _name;
    sym = _sym;

    if (sym.plist.size() == 0) {
        // Default value for TensorSym required, i.e. None in python
        std::vector<int> inds;
        for (unsigned int i = 0; i < indices.size(); i++) {
            inds.push_back(i);
        }
        sym = TensorSym({inds}, {1});
    }
}

bool operator==(const Tensor &a, const Tensor &b) {
    bool out;

    if (a.name != b.name) {
        out = false;
    }
    else {
        out = true;
        for (unsigned int i = 0; i < a.indices.size(); i++) {
            if (a.indices[i] != b.indices[i]) {
                out = false;
                break;
            }
        }
    }

    return out;
}

bool operator!=(const Tensor &a, const Tensor &b) {
    return (!(a == b));
}

bool operator<(const Tensor &a, const Tensor &b) {
    bool out = false;

    if (a.indices.size() < b.indices.size()) {
        out = true;
    }
    else if (a.indices.size() == b.indices.size()) {
        if (a.name < b.name) {  // FIXME is this ok?
            out = true;
        }
        else if (a.name == b.name) {
            for (unsigned int i = 0; i < a.indices.size(); i++) {
                if (a.indices[i] != b.indices[i]) {
                    out = a.indices[i] < b.indices[i];
                    break;
                }
            }
        }
        else {
            out = false;
        }
    }
    else{
        out = false;
    }

    return out;
}

bool operator<=(const Tensor &a, const Tensor &b) {
    return ((a < b) || (a == b));
}

bool operator>(const Tensor &a, const Tensor &b) {
    return (!(a <= b));
}

bool operator>=(const Tensor &a, const Tensor &b) {
    return (!(a < b));
}

std::string Tensor::repr() {
    std::string out = "";

    for (unsigned int i = 0; i < indices.size(); i++) {
        out = out + std::to_string(indices[i].index);
    }

    out = name + "_{" + out + "}";

    return out;
}

Tensor Tensor::_inc(int i) {
    std::vector<Idx> new_indices;

    for (unsigned int j = 0; j < indices.size(); j++) {
        Idx new_idx(indices[j].index + i, indices[j].space, indices[j].fermion);
        new_indices.push_back(new_idx);
    }

    return Tensor(new_indices, name, sym);
}

std::vector<Idx> Tensor::ilist() {
    std::vector<Idx> out;

    for (unsigned int i = 0; i < indices.size(); i++) {
        bool found = false;
        for (unsigned int j = 0; j < out.size(); j++) {
            if (indices[i] == out[j]) {
                found = true;
                break;
            }
        }

        if (!(found)) {
            out.push_back(indices[i]);
        }
    }

    return out;
}

void Tensor::transpose(std::vector<int> perm) {
    assert(perm.size() == indices.size());

    std::vector<Idx> new_indices;

    for (unsigned int i = 0; i < perm.size(); i++) {
        new_indices.push_back(indices[i]);
    }

    indices = new_indices;
}

Tensor Tensor::copy() {
    std::vector<Idx> new_indices;

    for (unsigned int i = 0; i < indices.size(); i++) {
        new_indices.push_back(idx_copy(indices[i]));
    }

    return Tensor(new_indices, name, sym);
}

Tensor permute(Tensor t, std::vector<int> p) {
    std::vector<Idx> permuted_indices;

    for (auto i = p.begin(); i < p.end(); i++) {
        permuted_indices.push_back(t.indices[*i]);
    }

    return Tensor(permuted_indices, t.name, t.sym);
}


// Sigma

Sigma::Sigma() {};

Sigma::Sigma(Idx _idx) {
    idx = _idx;
}

Sigma Sigma::_inc(int i) {
    Idx new_idx(idx.index + i, idx.space, idx.fermion);
    return Sigma(new_idx);
}

std::string Sigma::repr() {
    return "\\sum_{" + std::to_string(idx.index) + "}";
}

Sigma Sigma::copy() {
    return Sigma(idx_copy(idx));
}

bool operator==(const Sigma &a, const Sigma &b) {
    return (a.idx == b.idx);
}

bool operator!=(const Sigma &a, const Sigma &b) {
    return (!(a == b));
}

bool operator<(const Sigma &a, const Sigma &b) {
    return (a.idx < b.idx);
}

bool operator>(const Sigma &a, const Sigma &b) {
    return (a.idx > b.idx);
}

bool operator<=(const Sigma &a, const Sigma &b) {
    return (a.idx <= b.idx);
}

bool operator>=(const Sigma &a, const Sigma &b) {
    return (a.idx >= b.idx);
}


// Delta

Delta::Delta() {};

Delta::Delta(Idx _i1, Idx _i2) {
    i1 = _i1;
    i2 = _i2;
}

std::string Delta::repr() {
    return "\\delta_{" + std::to_string(i1.index) + "," + std::to_string(i2.index) + "}";
}

Delta Delta::_inc(int i) {
    Idx new_i1(i1.index + i, i1.space, i1.fermion);
    Idx new_i2(i2.index + i, i2.space, i2.fermion);

    return Delta(new_i1, new_i2);
}

Delta Delta::copy() {
    return Delta(idx_copy(i1), idx_copy(i2));
}

bool operator==(const Delta &a, const Delta &b) {
    return (((a.i1 == b.i1) && (a.i2 == b.i2)) || ((a.i1 == b.i2) && (a.i2 == b.i1)));
}

bool operator!=(const Delta &a, const Delta &b) {
    return (!(a == b));
}

Tensor tensor_from_delta(Delta d) {
    TensorSym sym({{0, 1}, {1, 0}}, {1, 1});
    return Tensor({d.i1, d.i2}, "delta", sym);
}

bool is_normal_ordered(std::vector<Operator> operators) {  // TODO: occ keyword
    int fa = -1;
    bool out = true;

    for (unsigned int i = 0; i < operators.size(); i++) {
        if ((fa == -1) && (!(operators[i].qp_creation()))) {
            fa = i;
        }
        if ((fa != -1) && (operators[i].qp_creation())) {
            out = false;
            break;
        }
    }

    return out;
}

std::pair<std::vector<Operator>, int> normal_ordered(std::vector<Operator> operators, int sign)
{
    // TODO: occ keyword
    if (is_normal_ordered(operators)) {
        std::pair<std::vector<Operator>, int> p;
        p.first = operators;
        p.second = sign;
        return p;
    }

    // Shouldn't be able to get here for a BOperator
    for (unsigned int i = 0; i < operators.size(); i++) {
        assert(operators[i].fermion);
    }

    int fa = -1;
    int swap = -1;

    // Can only get here for an FOperator
    for (unsigned int i = 0; i < operators.size(); i++) {
        if ((fa == -1) && (!(operators[i].qp_creation()))) {
            fa = i;
        }
        if ((fa != -1) && (operators[i].qp_creation())) {
            swap = i;
            break;
        }
    }

    assert(swap != -1);

    std::vector<Operator> newops;
    for (int i = 0; i < fa; i++) {
        newops.push_back(operators[i]);
    }
    newops.push_back(operators[swap]);
    for (int i = fa; i < swap; i++) {
        newops.push_back(operators[i]);
    }
    for (unsigned int i = swap+1; i < operators.size(); i++) {
        newops.push_back(operators[i]);
    }

    int newsign;
    if ((swap - fa) % 2 == 0) {
        newsign = 1;
    } else {
        newsign = -1;
    }

    sign = sign * newsign;

    return normal_ordered(newops, sign);
}
