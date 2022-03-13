/*
 *  Expressions
 */

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include<unordered_set>
#include<unordered_map>
#include<vector>
#include<numeric>
#include<algorithm>

#include "index.h"
#include "operator.h"

// proper grim
#define TERM_MAP_DATA_TYPE \
        std::pair< \
            std::string, \
            std::vector< \
                std::pair< \
                    std::string, \
                    std::unordered_set< \
                        std::string \
                    > \
                > \
            > \
        > \


// Create a struct for elements of TermMap to implement the equality
// and hashing functions
//struct TermMapElement {
//    std::string first;
//    std::vector<std::pair<std::string, std::unordered_set<std::string>>> second;
//
//    TermMapElement() { }
//    TermMapElement(
//            std::string first,
//            std::vector<std::pair<std::string, std::unordered_set<std::string>>> second
//    ) {
//        this->x = x;
//        this->y = y;
//    }
//
//    bool operator==(const TermMapElement &other) const 
//        // FIXME single breakpoint plz
//
//        if (this->first != other.first) {
//            return false;
//        } else {
//            if ((this->second).size() != (other.second.size())) {
//                return false;
//            } else {
//                for (int x = 0; x < other.second.size(); x++) {
//                    if ((this->second)[x].first != (other.second[x].first)) {
//                        return false;
//                    } else {
//                        if ((this->second)[x].second != (other.second[x].second)) {
//                            return false;
//                        }
//                    }
//                }
//            }
//        }
//
//        return true;
//    }
//}

class TermMap {
    public:
        std::vector<TERM_MAP_DATA_TYPE> data;  // TODO this needs to be a set

        void init(std::vector<Sigma> sums, std::vector<Tensor> tensors);
};

bool operator==(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b);
bool operator!=(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b);

bool operator==(const TermMap &a, const TermMap &b);
bool operator!=(const TermMap &a, const TermMap &b);

std::unordered_map<std::string, std::string> default_index_key();

int get_case(Delta dd, std::unordered_set<Idx, IdxHash> ilist);
void _resolve(
        std::vector<Sigma> &sums,
        std::vector<Tensor> &tensors,
        std::vector<Operator> &operators,
        std::vector<Delta> &deltas);


class Term {
    public:
        double scalar;  // TODO store as fraction
        std::vector<Sigma> sums;
        std::vector<Tensor> tensors;
        std::vector<Operator> operators;
        std::vector<Delta> deltas;
        std::unordered_map<std::string, std::string> index_key;

        void init(
                double _scalar,
                std::vector<Sigma> _sums,
                std::vector<Tensor> _tensors,
                std::vector<Operator> _operators,
                std::vector<Delta> _deltas,
                std::unordered_map<std::string, std::string> _index_key);
        void resolve();
        std::string repr();
        std::string _print_str(); // TODO
        std::string _idx_map(); // TODO
        std::vector<Idx> ilist();
        Term _inc(int i);
        Term copy();
};

Term operator*(Term &a, Term &b);
Term operator*(Term &a, double &b);
Term operator*(double &a, Term &b);
Term operator*(Term &a, int &b);
Term operator*(int &a, Term &b);
bool operator==(const Term &a, const Term &b);
bool operator!=(const Term &a, const Term &b);

class ATerm {
    public:
        double scalar;  // TODO store as fraction
        std::vector<Sigma> sums;
        std::vector<Tensor> tensors;
        std::unordered_map<std::string, std::string> index_key;

        void init(
                double _scalar,
                std::vector<Sigma> _sums,
                std::vector<Tensor> _tensors,
                std::unordered_map<std::string, std::string> _index_key);
        void init(Term term);

        ATerm _inc(int i);
        std::string repr();
        std::string _print_str(); // TODO
        std::string _idx_map(); // TODO
        std::string _einsum_str(); // TODO
        bool match(ATerm other);
        int pmatch(ATerm other);
        std::vector<Idx> ilist();
        unsigned int nidx();
        void sort_tensors();
        void merge_external();
        bool connected();
        bool reducible();
        void transpose(std::vector<int> perm);
        ATerm copy();
};

ATerm operator*(ATerm &a, ATerm &b);
ATerm operator*(ATerm &a, double &b);
ATerm operator*(double &a, ATerm &b);
ATerm operator*(ATerm &a, int &b);
ATerm operator*(int &a, ATerm &b);
bool operator==(const ATerm &a, const ATerm &b);
bool operator!=(const ATerm &a, const ATerm &b);


class Expression {
    public:
        std::vector<Term> terms;
        const double tthresh = 1e-15;

        void init(std::vector<Term> _terms);
        void resolve();
        std::string repr();
        std::string _print_str(); // TODO
        bool are_operators();
};

Expression operator+(const Expression &a, const Expression &b);
Expression operator-(const Expression &a, const Expression &b);
Expression operator*(Expression &a, Expression &b);
Expression operator*(Expression &a, double &b);
Expression operator*(double &a, Expression &b);
Expression operator*(Expression &a, int &b);
Expression operator*(int &a, Expression &b);
bool operator==(const Expression &a, const Expression &b);
bool operator!=(const Expression &a, const Expression &b);


class AExpression {
    public:
        std::vector<ATerm> terms;
        const double tthresh = 1e-15;

        void init();
        void init(std::vector<ATerm> _terms, bool _simplify, bool _sort);
        void init(Expression ex, bool _simplify, bool _sort);
        void simplify();
        void sort();
        void sort_tensors();
        std::string _print_str(); // TODO
        std::string _print_einsum(); // TODO
        bool connected();
        AExpression get_connected(bool _simplify, bool _sort);
        bool pmatch(AExpression other);
        void transpose(std::vector<int> perm);
};


#endif
