/*
 *  Wick theorem
 */

#ifndef QWICK_SRC_WICK_H_
#define QWICK_SRC_WICK_H_

#include <utility>
#include <vector>
#include <omp.h>

#include "operator.h"
#include "expression.h"

bool valid_contraction(const Operator &o1, const Operator & o2);
std::vector<std::vector<std::pair<Operator, Operator>>> pair_list(const std::vector<Operator> &lst);
std::pair<int, int> find_pair(const int i, const std::vector<std::pair<int, int>> &ipairs);
int get_sign(std::vector<std::pair<int, int>> &ipairs);
std::vector<std::vector<Operator>> split_operators(std::vector<Operator> &ops);
Expression apply_wick(Expression e);

struct OListsHash {
    std::size_t operator()(std::vector<std::vector<Operator>> a) const {
        std::size_t seed = 0;
        for (auto b = a.begin(); b != a.end(); b++) {
            for (auto c = (*b).begin(); c != (*b).end(); c++) {
                seed += OperatorHash()((*c));
            }
        }
        return seed;
    }
};


#endif  // QWICK_SRC_WICK_H_
