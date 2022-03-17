/*
 *  Operators
 */

#ifndef OPERATOR_H
#define OPERATOR_H

#include<string>
#include<vector>
#include<cassert>

#include "index.h"

class Operator{
    public:
        Idx idx;
        bool ca;
        bool fermion;
        bool projector;

        Operator();
        Operator(const Idx _idx, bool _ca, bool _fermion=true);

        std::string repr();
        std::string _print_str();  // TODO
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
        TensorSym sym;  // TODO handle default value

        Tensor();
        Tensor(const std::vector<Idx> _indices, const std::string _name, const TensorSym _sym=TensorSym());

        void hash();  // TODO
        Tensor _inc(int i);
        std::vector<Idx> ilist();
        std::string repr();
        std::string _istr();  // TODO
        std::string _print_str();  // TODO
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

        void hash();  // TODO
        Sigma _inc(int i);
        std::string repr();
        std::string _print_str();  // TODO
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

        void hash();  // TODO
        std::string repr();
        std::string _print_str();
        Delta _inc(int i);
        Delta copy();
};

bool operator==(const Delta &a, const Delta &b);
bool operator!=(const Delta &a, const Delta &b);

Tensor tensor_from_delta(Delta d);
bool is_normal_ordered(std::vector<Operator> operators);
std::pair<std::vector<Operator>, int> normal_ordered(std::vector<Operator> operators, int sign=1);

#endif
