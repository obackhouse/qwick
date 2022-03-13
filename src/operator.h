/*
 *  Operators
 */

#ifndef OPERATOR_H
#define OPERATOR_H

#include<string>
#include<vector>
#include<cassert>

#include "index.h"

// TODO better inheritence

class Operator{
    public:
        Idx idx;
        bool ca;
        bool fermion;
        bool projector;
        Operator copy() { assert(0); }
        std::string repr() { assert(0); }
        Operator _inc(int i) { assert(0); }
};

bool operator==(const Operator &a, const Operator &b);
bool operator!=(const Operator &a, const Operator &b);
Operator operator*(Operator &a, Operator &b);
Operator operator*(Operator &a, double &b);
Operator operator*(double &a, Operator &b);

class Projector: public Operator {
    public:
        bool projector = true;
        void init(const Idx _idx);
        std::string repr();
        std::string _print_str();  // TODO
        Projector _inc(int i);
        Projector dagger();
        Projector copy();
};

bool operator==(const Projector &a, const Projector &b);
bool operator!=(const Projector &a, const Projector &b);

class FOperator: public Operator {
    public:
        bool fermion = true;
        bool projector = false;

        void init(const Idx _idx, bool _ca);
        std::string repr();
        std::string _print_str();
        FOperator _inc(int i);
        FOperator dagger();
        FOperator copy();
        bool qp_creation();  // TODO need occ keyword?
        bool qp_annihilation();
};

class BOperator: public Operator {
    public:
        bool fermion = false;
        bool projector = false;

        void init(const Idx _idx, bool _ca);
        std::string repr();
        std::string _print_str();  // TODO
        BOperator _inc(int i);
        BOperator dagger();
        BOperator copy();
        bool qp_creation();  // TODO need occ keyword?
        bool qp_annihilation();
};

class TensorSym {
    public:
        std::vector<std::vector<int>> plist;
        std::vector<int> signs;
        std::vector<std::pair<std::vector<int>, int>> tlist;

        void init(const std::vector<std::vector<int>> _plist, std::vector<int> _signs);
};

class Tensor {
    public:
        std::vector<Idx> indices;
        std::string name;
        TensorSym sym;  // TODO handle default value

        void init(const std::vector<Idx> _indices, const std::string _name, const TensorSym _sym);
        void hash();  // TODO
        Tensor _inc(int i);
        std::vector<Idx> ilist();
        std::string repr();  // TODO
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

        void init(Idx _idx);
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

        void init(Idx _i1, Idx _i2);
        void hash();  // TODO
        std::string repr();
        std::string _print_str();
        Delta _inc(int i);
        Delta copy();
};

bool operator==(const Delta &a, const Delta &b);
bool operator!=(const Delta &a, const Delta &b);

Tensor tensor_from_delta(Delta d);
bool is_normal_ordered(std::vector<FOperator> operators);
bool is_normal_ordered(std::vector<BOperator> operators);
std::pair<std::vector<FOperator>, int> normal_ordered(std::vector<FOperator> operators, int sign);
std::pair<std::vector<BOperator>, int> normal_ordered(std::vector<BOperator> operators, int sign);

#endif
