/*
 *  Operators
 */

#ifndef OPERATOR_H
#define OPERATOR_H

#include "index.h"

#include <string>
#include <vector>
#include <cassert>
#include <unordered_set>
#include <unordered_map>

class Operator{
    public:
        Idx idx;
        bool ca;
        bool fermion;
        bool projector;

        Operator();
        Operator(const Idx _idx, bool _ca, bool _fermion=true);

        std::string repr();
        std::string _print_str(std::unordered_map<Idx, std::string, IdxHash> imap);
        Operator _inc(int i);
        Operator dagger();
        Operator copy();
        bool qp_creation();
        bool qp_annihilation();
};

bool operator==(const Operator &a, const Operator &b);
bool operator!=(const Operator &a, const Operator &b);

class TensorSym {
    public:
        std::vector<std::vector<int>> plist;
        std::vector<int> signs;
        std::vector<std::pair<std::vector<int>, int>> tlist;

        TensorSym();
        TensorSym(const std::vector<std::vector<int>> _plist, std::vector<int> _signs);
};

class Tensor {
    public:
        std::vector<Idx> indices;
        std::string name;
        TensorSym sym;

        Tensor();
        Tensor(const std::vector<Idx> _indices, const std::string _name, const TensorSym _sym=TensorSym());

        Tensor _inc(int i);
        std::vector<Idx> ilist();
        std::string repr();
        std::string _istr(std::unordered_map<Idx, std::string, IdxHash> imap);
        std::string _print_str(std::unordered_map<Idx, std::string, IdxHash> imap);
        void transpose(std::vector<int>);
        Tensor copy();
};

bool operator==(const Tensor &a, const Tensor &b);
bool operator!=(const Tensor &a, const Tensor &b);
bool operator<(const Tensor &a, const Tensor &b);
bool operator<=(const Tensor &a, const Tensor &b);
bool operator>(const Tensor &a, const Tensor &b);
bool operator>=(const Tensor &a, const Tensor &b);

Tensor permute(Tensor t, std::vector<int> p);

class Sigma {
    public:
        Idx idx;

        Sigma();
        Sigma(Idx _idx);

        Sigma _inc(int i);
        std::string repr();
        std::string _print_str(std::unordered_map<Idx, std::string, IdxHash> imap);
        Sigma copy();
};

bool operator==(const Sigma &a, const Sigma &b);
bool operator!=(const Sigma &a, const Sigma &b);
bool operator<(const Sigma &a, const Sigma &b);
bool operator>(const Sigma &a, const Sigma &b);
bool operator<=(const Sigma &a, const Sigma &b);
bool operator>=(const Sigma &a, const Sigma &b);

class Delta {
    public:
        Idx i1;
        Idx i2;

        Delta();
        Delta(Idx _i1, Idx _i2);

        std::string repr();
        std::string _print_str(std::unordered_map<Idx, std::string, IdxHash> imap);
        Delta _inc(int i);
        Delta copy();
};

bool operator==(const Delta &a, const Delta &b);
bool operator!=(const Delta &a, const Delta &b);

Tensor tensor_from_delta(Delta d);
bool is_normal_ordered(std::vector<Operator> operators);
std::pair<std::vector<Operator>, int> normal_ordered(std::vector<Operator> operators, int sign=1);

#endif
