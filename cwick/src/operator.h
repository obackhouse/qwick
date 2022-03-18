/*
 *  Operators
 */

#ifndef CWICK_SRC_OPERATOR_H_
#define CWICK_SRC_OPERATOR_H_

#include <string>
#include <vector>
#include <cassert>
#include <utility>
#include <unordered_set>
#include <unordered_map>

#include "index.h"


// Operator

class Operator{
 public:
    // Attributes
    Idx idx;
    bool ca;
    bool fermion;
    bool projector;

    // Constructors
    Operator();
    Operator(const Idx &_idx, const bool _ca, const bool _fermion = true);

    // Functions
    Operator _inc(const int i) const;
    Operator dagger() const;
    Operator copy() const;
    bool qp_creation() const;
    bool qp_annihilation() const;

    // String representations
    std::string repr() const;
    std::string _print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const;
};

bool operator==(const Operator &a, const Operator &b);
bool operator!=(const Operator &a, const Operator &b);


// Tensor

class TensorSym {
 public:
    // Attributes
    std::vector<std::vector<int>> plist;
    std::vector<int> signs;
    std::vector<std::pair<std::vector<int>, int>> tlist;

    // Constructors
    TensorSym();
    TensorSym(const std::vector<std::vector<int>> &plist, const std::vector<int> &signs);
};

class Tensor {
 public:
    // Attributes
    std::vector<Idx> indices;
    std::string name;
    TensorSym sym;

    // Constructors
    Tensor();
    Tensor(const std::vector<Idx> &indices,
           const std::string &name,
           const TensorSym &sym = TensorSym());

    // Functions
    Tensor _inc(const int i) const;
    std::vector<Idx> ilist() const;
    void transpose(const std::vector<int> &perm);
    Tensor copy() const;

    // String representations
    std::string repr() const;
    std::string _istr(const std::unordered_map<Idx, std::string, IdxHash> &imap) const;
    std::string _print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const;
};

bool operator==(const Tensor &a, const Tensor &b);
bool operator!=(const Tensor &a, const Tensor &b);
bool operator<(const Tensor &a, const Tensor &b);
bool operator<=(const Tensor &a, const Tensor &b);
bool operator>(const Tensor &a, const Tensor &b);
bool operator>=(const Tensor &a, const Tensor &b);

Tensor permute(const Tensor &t, const std::vector<int> &p);


// Sigma

class Sigma {
 public:
    // Attributes
    Idx idx;

    // Constructors
    Sigma();
    explicit Sigma(const Idx &idx);

    // Functions
    Sigma _inc(const int i) const;
    Sigma copy() const;

    // String representations
    std::string repr() const;
    std::string _print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const;
};

bool operator==(const Sigma &a, const Sigma &b);
bool operator!=(const Sigma &a, const Sigma &b);
bool operator<(const Sigma &a, const Sigma &b);
bool operator>(const Sigma &a, const Sigma &b);
bool operator<=(const Sigma &a, const Sigma &b);
bool operator>=(const Sigma &a, const Sigma &b);


// Delta

class Delta {
 public:
    // Attributes
    Idx i1;
    Idx i2;

    // Constructors
    Delta();
    Delta(const Idx &_i1, const Idx &_i2);

    // Functions
    Delta _inc(const int i) const;
    Delta copy() const;

    // String representations
    std::string repr() const;
    std::string _print_str(const std::unordered_map<Idx, std::string, IdxHash> &imap) const;
};

bool operator==(const Delta &a, const Delta &b);
bool operator!=(const Delta &a, const Delta &b);

Tensor tensor_from_delta(const Delta &d);
bool is_normal_ordered(const std::vector<Operator> &operators);
std::pair<std::vector<Operator>, int> normal_ordered(const std::vector<Operator> &operators, const int sign = 1);

#endif  // CWICK_SRC_OPERATOR_H_
