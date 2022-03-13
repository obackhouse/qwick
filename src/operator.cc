/*
 *  Operators
 */

#include "index.h"
#include "operator.h"


bool operator==(const Operator &a, const Operator &b) { assert(0); }
bool operator!=(const Operator &a, const Operator &b) { assert(0); }
Operator operator*(Operator &a, Operator &b) { assert(0); }
Operator operator*(Operator &a, double &b) { assert(0); }
Operator operator*(double &a, Operator &b) { assert(0); }


// Projector

void Projector::init(Idx _idx) {
    idx = _idx;
}

std::string repr() {
    return "P";
}

Projector Projector::_inc(int i) {
    return *this;
}

Projector Projector::dagger() {
    return *this;
}

Projector Projector::copy() {
    return *this;
}

bool operator==(const Projector &a, const Projector &b) {
    // true if both are Projector objects, will fail otherwise
    return true;
}

bool operator!=(const Projector &a, const Projector &b) {
    return !(a == b);
}


// FOperator

void FOperator::init(const Idx _idx, bool _ca) {
    assert(_idx.fermion);
    idx = _idx;
    ca = _ca;
}

std::string FOperator::repr() {
    std::string out;

    if (ca) {
        out = "a^{\\dagger}_" + idx.repr();
    } else {
        out = "a_" + idx.repr();
    }

    return out;
}

FOperator FOperator::_inc(int i) {
    Idx new_idx;
    new_idx.init(idx.index + i, idx.space, true);

    FOperator out;
    out.init(new_idx, ca);

    return out;
}

FOperator FOperator::dagger() {
    FOperator out;
    out.init(idx_copy(idx), !(ca));

    return out;
}

FOperator FOperator::copy() {
    FOperator out;
    out.init(idx_copy(idx), ca);

    return out;
}

bool FOperator::qp_creation() {
    bool out;

    if (!(is_occupied(idx)) && ca) {
        out = true;
    } else if (is_occupied(idx) and !(ca)) {
        out = true;
    } else {
        out = false;
    }

    return out;
}

bool FOperator::qp_annihilation() {
    return (!(qp_creation()));
}


// BOperator

void BOperator::init(const Idx _idx, bool _ca) {
    assert(!(_idx.fermion));
    idx = _idx;
    ca = _ca;
}

std::string BOperator::repr() {
    std::string out;

    if (ca) {
        out = "b^{\\dagger}_" + idx.repr();
    } else {
        out = "b_" + idx.repr();
    }

    return out;
}

BOperator BOperator::_inc(int i) {
    Idx new_idx;
    new_idx.init(idx.index + i, idx.space, false);

    BOperator out;
    out.init(new_idx, ca);

    return out;
}

BOperator BOperator::dagger() {
    BOperator out;
    out.init(idx_copy(idx), !(ca));

    return out;
}

BOperator BOperator::copy() {
    BOperator out;
    out.init(idx_copy(idx), ca);

    return out;
}

bool BOperator::qp_creation() {
    return ca;
}

bool BOperator::qp_annihilation() {
    return (!(qp_creation()));
}


// TensorSym

void TensorSym::init(
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

void Tensor::init(
        const std::vector<Idx> _indices,
        const std::string _name,
        const TensorSym _sym)
{
    indices = _indices;
    name = _name;
    sym = _sym;
}

bool operator==(const Tensor &a, const Tensor &b) {
    bool out;

    if (a.name != b.name) {
        out = false;
    } else {
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
    bool out;

    if (a.indices.size() < b.indices.size()) {
        out = true;
    } else if (a.indices.size() == b.indices.size()) {
        if (a.name < b.name) {  // FIXME is this ok?
            out = true;
        } else if (a.name == b.name) {
            for (unsigned int i = 0; i < a.indices.size(); i++) {
                if (a.indices[i] != b.indices[i]) {
                    out = a.indices[i] < b.indices[i];
                }
            }
        } else {
            out = false;
        }
    } else{
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
        out = out + indices[i].repr();
    }

    out = name + "_{" + out + "}";

    return out;
}

Tensor Tensor::_inc(int i) {
    std::vector<Idx> new_indices;

    for (unsigned int j = 0; j < indices.size(); j++) {
        Idx new_idx;
        new_idx.init(indices[j].index + i, indices[j].space, indices[j].fermion);

        new_indices.push_back(new_idx);
    }

    Tensor new_tensor;
    new_tensor.init(new_indices, name, sym);

    return new_tensor;
}

std::vector<Idx> Tensor::ilist() {
    std::vector<Idx> out;

    for (unsigned int i = 0; i < indices.size(); i++) {
        for (unsigned int j = 0; j < out.size(); j++) {
            bool found = false;

            if (indices[i] == out[j]) {
                found = true;
                break;
            }

            if (!(found)) {
                out.push_back(indices[i]);
            }
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

    Tensor new_tensor;
    new_tensor.init(new_indices, name, sym);

    return new_tensor;
}

Tensor permute(Tensor t, std::vector<int> p) {
    std::vector<Idx> permuted_indices;

    for (unsigned int i = 0; i < t.indices.size(); i++) {
        permuted_indices.push_back(t.indices[p[i]]);
    }

    Tensor new_tensor;
    new_tensor.init(permuted_indices, t.name, t.sym);

    return new_tensor;
}


// Sigma

void Sigma::init(Idx _idx) {
    idx = _idx;
}

Sigma Sigma::_inc(int i) {
    Idx new_idx;
    new_idx.init(idx.index + i, idx.space, idx.fermion);

    Sigma out;
    out.init(new_idx);

    return out;
}

std::string Sigma::repr() {
    return "\\sum_{" + std::to_string(idx.index) + "}";
}

Sigma Sigma::copy() {
    Sigma out;
    out.init(idx_copy(idx));

    return out;
}

bool operator==(const Sigma &a, const Sigma &b) {
    return (a.idx == b.idx);
}

bool operator!=(const Sigma &a, const Sigma &b) {
    return (!(a == b));
}

bool operator<(const Sigma &a, const Sigma &b) {
    return (a < b);
}

bool operator>(const Sigma &a, const Sigma &b) {
    return (a > b);
}

bool operator<=(const Sigma &a, const Sigma &b) {
    return (a <= b);
}

bool operator>=(const Sigma &a, const Sigma &b) {
    return (a >= b);
}


// Delta

void Delta::init(Idx _i1, Idx _i2) {
    i1 = _i1;
    i2 = _i2;
}

std::string Delta::repr() {
    return "\\delta_{" + std::to_string(i1.index) + std::to_string(i2.index) + "}";
}

Delta Delta::_inc(int i) {
    Idx new_i1;
    new_i1.init(i1.index + i, i1.space, i1.fermion);

    Idx new_i2;
    new_i2.init(i2.index + i, i2.space, i2.fermion);

    Delta out;
    out.init(new_i1, new_i2);

    return out;
}

Delta Delta::copy() {
    Delta out;
    out.init(idx_copy(i1), idx_copy(i2));

    return out;
}

bool operator==(const Delta &a, const Delta &b) {
    return (((a.i1 == b.i1) && (a.i2 == b.i2)) || ((a.i1 == b.i2) && (a.i2 == b.i1)));
}

bool operator!=(const Delta &a, const Delta &b) {
    return (!(a == b));
}

Tensor tensor_from_delta(Delta d) {
    TensorSym sym;
    sym.init({{0, 1}, {1, 0}}, {1, 1});

    Tensor t;
    t.init({d.i1, d.i2}, "delta", sym);

    return t;
}

bool is_normal_ordered(std::vector<FOperator> operators) {  // TODO: occ keyword
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

bool is_normal_ordered(std::vector<BOperator> operators) {  // TODO: occ keyword
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

std::pair<std::vector<FOperator>, int> normal_ordered(
        std::vector<FOperator> operators,
        int sign = 1)
{
    // TODO: occ keyword
    if (is_normal_ordered(operators)) {
        std::pair<std::vector<FOperator>, int> p;
        p.first = operators;
        p.second = sign;
        return p;
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

    std::vector<FOperator> newops;
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

std::pair<std::vector<BOperator>, int> normal_ordered(
        std::vector<BOperator> operators,
        int sign = 1)
{
    // TODO: occ keyword
    if (is_normal_ordered(operators)) {
        std::pair<std::vector<BOperator>, int> p;
        p.first = operators;
        p.second = sign;
        return p;
    }

    // Shouldn't be able to get here for a BOperator
    assert(false);
}
