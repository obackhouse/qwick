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

using TERM_MAP_SUB_DATA_TYPE = std::pair<std::string, std::unordered_set<std::string>>;

// FIXME clean hashes
struct TermMapSubDataHash {
    std::size_t operator()(TERM_MAP_SUB_DATA_TYPE a) const {
        std::size_t seed = std::hash<std::string>()(a.first);
        for (auto b = a.second.begin(); b != a.second.end(); b++) {
            // FIXME: these must commute!!! will addition result in hash collision?
            //seed ^= std::hash<std::string>()((*b)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed += std::hash<std::string>()((*b));
        }
        return seed;
    }
};

using TERM_MAP_DATA_TYPE = std::pair<std::string, std::unordered_set<TERM_MAP_SUB_DATA_TYPE, TermMapSubDataHash>>;

struct TermMapDataHash {
    std::size_t operator()(TERM_MAP_DATA_TYPE a) const {
        std::size_t seed = std::hash<std::string>()(a.first);
        for (auto b = a.second.begin(); b != a.second.end(); b++) {
            // FIXME: these must commute!!! will addition result in hash collision?
            //seed ^= TermMapSubDataHash()((*b)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed += TermMapSubDataHash()((*b));
        }
        return seed;
    }
};

class TermMap {
    public:
        std::unordered_set<TERM_MAP_DATA_TYPE, TermMapDataHash> data;  // TODO this needs to be a set

        TermMap();
        TermMap(std::vector<Sigma> sums, std::vector<Tensor> tensors);
};

struct TermMapHash {
    std::size_t operator()(TermMap a) const {
        std::size_t seed = 0;
        for (auto b = a.data.begin(); b != a.data.end(); b++) {
            // FIXME: these must commute!!! will addition result in hash collision?
            //seed ^= TermMapDataHash()(*b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed += TermMapDataHash()(*b);
        }
        return seed;
    }
};

bool operator==(TERM_MAP_SUB_DATA_TYPE &a, TERM_MAP_SUB_DATA_TYPE &b);
bool operator!=(TERM_MAP_SUB_DATA_TYPE &a, TERM_MAP_SUB_DATA_TYPE &b);

bool operator==(TERM_MAP_DATA_TYPE &a, TERM_MAP_DATA_TYPE &b);
bool operator!=(TERM_MAP_DATA_TYPE &a, TERM_MAP_DATA_TYPE &b);

bool operator==(TermMap &a, TermMap &b);
bool operator!=(TermMap &a, TermMap &b);

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

        Term();
        Term(
                double _scalar,
                std::vector<Sigma> _sums,
                std::vector<Tensor> _tensors,
                std::vector<Operator> _operators,
                std::vector<Delta> _deltas,
                std::unordered_map<std::string, std::string> _index_key=default_index_key()
        );

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

        ATerm();
        ATerm(
                double _scalar,
                std::vector<Sigma> _sums,
                std::vector<Tensor> _tensors,
                std::unordered_map<std::string, std::string> _index_key=default_index_key()
        );
        ATerm(Term term);

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
bool operator<(const ATerm &a, const ATerm &b);
bool operator<=(const ATerm &a, const ATerm &b);
bool operator>(const ATerm &a, const ATerm &b);
bool operator>=(const ATerm &a, const ATerm &b);


class Expression {
    public:
        std::vector<Term> terms;
        const double tthresh = 1e-15;

        Expression();
        Expression(std::vector<Term> _terms);

        void resolve();
        std::string repr();
        std::string _print_str(); // TODO
        bool are_operators();
};

Expression operator+(const Expression &a, const Expression &b);
Expression operator-(Expression &a, Expression &b);
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

        AExpression();
        AExpression(std::vector<ATerm> _terms, bool _simplify=true, bool _sort=true);
        AExpression(Expression ex, bool _simplify=true, bool _sort=true);

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
