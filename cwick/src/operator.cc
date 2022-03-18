/*
 *  Operators
 */

#include "util.h"
#include "index.h"
#include "operator.h"


Operator::Operator() {
    // Projector
    projector = true;
}

Operator::Operator(const Idx &_idx, const bool _ca, const bool _fermion) {
    // FOperator or BOperator
    idx = _idx;
    ca = _ca;
    projector = false;
    fermion = _fermion;
}

std::string Operator::repr() const {
    if (projector) {
        return "P";
    } else if (fermion) {
        if (ca) {
            return "a^{\\dagger}_" + idx.repr();
        } else {
            return "a_" + idx.repr();
        }
    } else {
        if (ca) {
            return "b^{\\dagger}_" + idx.repr();
        } else {
            return "b_" + idx.repr();
        }
    }
}

std::string Operator::_print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const {
    if (projector) {
        return "P";
    } else if (fermion) {
        if (ca) {
            return "a^{\\dagger}_" + imap.at(idx);
        } else {
            return "a_" + imap.at(idx);
        }
    } else {
        if (ca) {
            return "b^{\\dagger}_" + imap.at(idx);
        } else {
            return "b_" + imap.at(idx);
        }
    }
}

Operator Operator::_inc(const int i) const {
    if (projector) {
        return *this;
    } else {
        Idx new_idx(idx.index + i, idx.space, fermion);
        return Operator(new_idx, ca, fermion);
    }
}

Operator Operator::dagger() const {
    if (projector) {
        return *this;
    } else {
        return Operator(idx_copy(idx), !(ca), fermion);
    }
}

Operator Operator::copy() const {
    if (projector) {
        return *this;
    } else {
        return Operator(idx_copy(idx), ca, fermion);
    }
}

bool operator==(const Operator &a, const Operator &b) {
    if (a.projector || b.projector) {
        return (a.projector && b.projector);
    } else {
        return ((a.fermion == b.fermion) && (a.idx == b.idx) && (a.ca == b.ca));
    }
}

bool operator!=(const Operator &a, const Operator &b) {
    return (!(a == b));
}

bool Operator::qp_creation() const {
    assert(!(projector));

    if (fermion) {
        if (!(is_occupied(idx)) && ca) {
            return true;
        } else if (is_occupied(idx) && (!(ca))) {
            return true;
        } else {
            return false;
        }
    } else {
        return ca;
    }
}

bool Operator::qp_annihilation() const {
    return (!(qp_creation()));
}


// TensorSym

TensorSym::TensorSym() {}

TensorSym::TensorSym(const std::vector<std::vector<int>> &_plist,
                     const std::vector<int> &_signs) {
    plist = _plist;
    signs = _signs;

    assert(_plist.size() == _signs.size());
    tlist.reserve(plist.size());
    for (unsigned int i = 0; i < plist.size(); i++) {
        std::pair<std::vector<int>, int> p(plist[i], signs[i]);
        tlist.push_back(p);
    }
}


// Tensor

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<Idx> &_indices,
               const std::string &_name,
               const TensorSym &_sym) {
    indices = _indices;
    name = _name;
    sym = _sym;

    if (sym.plist.size() == 0) {
        std::vector<std::vector<int>> inds(1);
        std::vector<int> signs = {1};
        for (unsigned int i = 0; i < indices.size(); i++) {
            inds[0].push_back(i);
        }
        sym = TensorSym(inds, signs);
    }
}

bool operator==(const Tensor &a, const Tensor &b) {
    if (a.name != b.name) {
        return false;
    } else {
        for (unsigned int i = 0; i < a.indices.size(); i++) {
            if (a.indices[i] != b.indices[i]) {
                return false;
            }
        }
        return true;
    }
}

bool operator!=(const Tensor &a, const Tensor &b) {
    return (!(a == b));
}

bool operator<(const Tensor &a, const Tensor &b) {
    if (a.indices.size() < b.indices.size()) {
        return true;
    } else if (a.indices.size() == b.indices.size()) {
        if (a.name < b.name) {
            return true;
        } else if (a.name == b.name) {
            for (unsigned int i = 0; i < a.indices.size(); i++) {
                if (a.indices[i] != b.indices[i]) {
                    return a.indices[i] < b.indices[i];
                }
            }
            return false;
        } else {
            return false;
        }
    } else {
        return false;
    }
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

std::string Tensor::repr() const {
    std::string out = "";
    for (auto i = indices.begin(); i < indices.end(); i++) {
        out += std::to_string((*i).index);
    }

    return name + "_{" + out + "}";
}

std::string Tensor::_istr(const std::unordered_map<Idx, std::string, IdxHash> &imap) const {
    std::string out = "";
    for (auto i = indices.begin(); i < indices.end(); i++) {
        out += imap.at(*i);
    }

    return out;
}

std::string Tensor::_print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const {
    if (name == "") {
        return "";
    }

    std::string out = "";
    for (auto i = indices.begin(); i < indices.end(); i++) {
        out += imap.at(*i);
    }

    return name + "_{" + out + "}";
}

Tensor Tensor::_inc(const int i) const {
    std::vector<Idx> new_indices(indices.size());

    for (unsigned int j = 0; j < indices.size(); j++) {
        Idx new_idx(indices[j].index + i, indices[j].space, indices[j].fermion);
        new_indices[j] = new_idx;
    }

    return Tensor(new_indices, name, sym);
}

std::vector<Idx> Tensor::ilist() const {
    std::vector<Idx> out;

    for (auto i = indices.begin(); i < indices.end(); i++) {
        bool found = false;
        for (auto j = out.begin(); j < out.end(); j++) {
            if ((*i) == (*j)) {
                found = true;
                break;
            }
        }

        if (!(found)) {
            out.push_back(*i);
        }
    }

    return out;
}

void Tensor::transpose(const std::vector<int> &perm) {
    std::vector<Idx> new_indices(indices.size());

    assert(perm.size() == indices.size());
    for (unsigned int i = 0; i < perm.size(); i++) {
        new_indices[i] = indices[perm[i]];
    }

    indices = new_indices;
}

Tensor Tensor::copy() const {
    std::vector<Idx> new_indices(indices.size());

    for (unsigned int i = 0; i < indices.size(); i++) {
        new_indices[i] = idx_copy(indices[i]);
    }

    return Tensor(new_indices, name, sym);
}

Tensor permute(const Tensor &t, const std::vector<int> &p) {
    std::vector<Idx> permuted_indices(p.size());

    for (unsigned int i = 0; i < p.size(); i++) {
        permuted_indices[i] = t.indices[p[i]];
    }

    return Tensor(permuted_indices, t.name, t.sym);
}


// Sigma

Sigma::Sigma() {}

Sigma::Sigma(const Idx &_idx) {
    idx = _idx;
}

Sigma Sigma::_inc(const int i) const {
    Idx new_idx(idx.index + i, idx.space, idx.fermion);
    return Sigma(new_idx);
}

std::string Sigma::repr() const {
    return "\\sum_{" + std::to_string(idx.index) + "}";
}

std::string Sigma::_print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const {
    return "\\sum_{" + imap.at(idx) + "}";
}

Sigma Sigma::copy() const {
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

Delta::Delta() {}

Delta::Delta(const Idx &_i1, const Idx &_i2) {
    i1 = _i1;
    i2 = _i2;
}

std::string Delta::repr() const {
    return "\\delta_{" + std::to_string(i1.index) + "," + std::to_string(i2.index) + "}";
}

std::string Delta::_print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const {
    return "\\delta_{" + imap.at(i1) + imap.at(i2) + "}";
}

Delta Delta::_inc(const int i) const {
    Idx new_i1(i1.index + i, i1.space, i1.fermion);
    Idx new_i2(i2.index + i, i2.space, i2.fermion);
    return Delta(new_i1, new_i2);
}

Delta Delta::copy() const {
    return Delta(idx_copy(i1), idx_copy(i2));
}

bool operator==(const Delta &a, const Delta &b) {
    return (((a.i1 == b.i1) && (a.i2 == b.i2)) || ((a.i1 == b.i2) && (a.i2 == b.i1)));
}

bool operator!=(const Delta &a, const Delta &b) {
    return (!(a == b));
}

Tensor tensor_from_delta(const Delta &d) {
    std::vector<std::vector<int>> inds = {{0, 1}, {1, 0}};
    std::vector<int> signs = {1, 1};
    TensorSym sym(inds, signs);

    std::vector<Idx> indices = {d.i1, d.i2};

    return Tensor(indices, "delta", sym);
}

bool is_normal_ordered(const std::vector<Operator> &operators) {  // TODO: occ keyword
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

std::pair<std::vector<Operator>, int> normal_ordered(const std::vector<Operator> &operators, const int sign) {
    // TODO: occ keyword
    if (is_normal_ordered(operators)) {
        std::pair<std::vector<Operator>, int> p(operators, sign);
        return p;
    }

    // Shouldn't be able to get here for a BOperator
    for (auto op = operators.begin(); op < operators.end(); op++) {
        assert((*op).fermion);
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

    int newsign = ((swap - fa) % 2 == 0) ? 1 : -1;

    return normal_ordered(newops, sign * newsign);
}
