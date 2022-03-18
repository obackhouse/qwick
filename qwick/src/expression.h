/*
 *  Expressions
 */

#ifndef QWICK_SRC_EXPRESSION_H_
#define QWICK_SRC_EXPRESSION_H_

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <string>

#include "index.h"
#include "operator.h"


// TermMap

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
    // Attributes
    std::unordered_set<TERM_MAP_DATA_TYPE, TermMapDataHash> data;

    // Constructors
    TermMap();
    TermMap(const std::vector<Sigma> &sums, const std::vector<Tensor> &tensors);
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

bool operator==(const TERM_MAP_SUB_DATA_TYPE &a, const TERM_MAP_SUB_DATA_TYPE &b);
bool operator!=(const TERM_MAP_SUB_DATA_TYPE &a, const TERM_MAP_SUB_DATA_TYPE &b);

bool operator==(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b);
bool operator!=(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b);

bool operator==(const TermMap &a, const TermMap &b);
bool operator!=(const TermMap &a, const TermMap &b);

extern const std::unordered_map<std::string, std::string> default_index_key;

int get_case(const Delta &dd, const std::unordered_set<Idx, IdxHash> &ilist);

void _resolve(std::vector<Sigma> &sums,
              std::vector<Tensor> &tensors,
              std::vector<Operator> &operators,
              std::vector<Delta> &deltas);


// Term

class Term {
 public:
    // Attributes
    double scalar;
    std::vector<Sigma> sums;
    std::vector<Tensor> tensors;
    std::vector<Operator> operators;
    std::vector<Delta> deltas;
    std::unordered_map<std::string, std::string> index_key;

    // Constructors
    Term();
    Term(const double scalar,
         const std::vector<Sigma> &sums,
         const std::vector<Tensor> &tensors,
         const std::vector<Operator> &operators,
         const std::vector<Delta> &deltas,
         const std::unordered_map<std::string, std::string> index_key=default_index_key);

    // Functions
    void resolve();
    std::unordered_map<Idx, std::string, IdxHash> _idx_map() const;
    std::vector<Idx> ilist() const;
    Term _inc(int i) const;
    Term copy() const;

    // String representations
    std::string repr() const;
    std::string _print_str(const bool with_scalar = true) const;
};

Term operator*(const Term &a, const Term &b);
Term operator*(const Term &a, const double &b);
Term operator*(const double &a, const Term &b);
Term operator*(const Term &a, const int &b);
Term operator*(const int &a, const Term &b);
bool operator==(const Term &a, const Term &b);
bool operator!=(const Term &a, const Term &b);


// ATerm

class ATerm {
 public:
    // Attributes
    double scalar;
    std::vector<Sigma> sums;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, std::string> index_key;

    // Constructors
    ATerm();
    ATerm(
            const double _scalar,
            const std::vector<Sigma> &_sums,
            const std::vector<Tensor> &_tensors,
            const std::unordered_map<std::string, std::string> _index_key=default_index_key
    );
    explicit ATerm(const Term &term);

    // Functions
    ATerm _inc(const int i) const;
    std::unordered_map<Idx, std::string, IdxHash> _idx_map() const;
    bool match(const ATerm other) const;
    int pmatch(const ATerm other) const;
    std::vector<Idx> ilist() const;
    unsigned int nidx() const;
    void sort_tensors();
    void merge_external();
    bool connected() const;
    bool reducible() const;
    void transpose(const std::vector<int> &perm);
    ATerm copy() const;

    // String representations
    std::string repr() const;
    std::string _print_str(const bool with_scalar = true) const;
    std::string _einsum_str() const;
};

ATerm operator*(const ATerm &a, const ATerm &b);
ATerm operator*(const ATerm &a, const double &b);
ATerm operator*(const double &a, const ATerm &b);
ATerm operator*(const ATerm &a, const int &b);
ATerm operator*(const int &a, const ATerm &b);
bool operator==(const ATerm &a, const ATerm &b);
bool operator!=(const ATerm &a, const ATerm &b);
bool operator<(const ATerm &a, const ATerm &b);
bool operator<=(const ATerm &a, const ATerm &b);
bool operator>(const ATerm &a, const ATerm &b);
bool operator>=(const ATerm &a, const ATerm &b);


// Expression

class Expression {
 public:
    // Attributes
    std::vector<Term> terms;
    const double tthresh = 1e-15;

    // Constructors
    Expression();
    explicit Expression(const std::vector<Term> &terms);

    // Functions
    void resolve();
    bool are_operators() const;

    // String representations
    std::string repr() const;
    std::string _print_str() const;
};

Expression operator+(const Expression &a, const Expression &b);
Expression operator-(const Expression &a, const Expression &b);
Expression operator*(const Expression &a, const Expression &b);
Expression operator*(const Expression &a, const double &b);
Expression operator*(const double &a, const Expression &b);
Expression operator*(const Expression &a, const int &b);
Expression operator*(const int &a, const Expression &b);
bool operator==(const Expression &a, const Expression &b);
bool operator!=(const Expression &a, const Expression &b);


// AExpression

class AExpression {
 public:
    // Attributes
    std::vector<ATerm> terms;
    const double tthresh = 1e-15;

    // Constructors
    AExpression();
    AExpression(const std::vector<ATerm> &terms, const bool simplify = true, const bool sort = true);
    AExpression(const Expression &ex, const bool simplify = true, const bool sort = true);

    // Functions
    void simplify();
    void sort();
    void sort_tensors();
    bool connected() const;
    AExpression get_connected(const bool _simplify = true, const bool _sort = true) const;
    bool pmatch(const AExpression &other) const;
    void transpose(const std::vector<int> &perm);

    // String representations
    std::string _print_str() const;
    std::string _print_einsum(const std::string lhs="") const;
};


#endif  // QWICK_SRC_EXPRESSION_H_
