/*
 *  Wick theorem
 */

#ifndef CWICK_SRC_WICK_H_
#define CWICK_SRC_WICK_H_

#include <utility>
#include <vector>

#include "operator.h"
#include "expression.h"

bool valid_contraction(const Operator &o1, const Operator & o2);
std::vector<std::vector<std::pair<Operator, Operator>>> pair_list(const std::vector<Operator> &lst);
std::pair<int, int> find_pair(const int i, const std::vector<std::pair<int, int>> &ipairs);
int get_sign(std::vector<std::pair<int, int>> &ipairs);
std::vector<std::vector<Operator>> split_operators(std::vector<Operator> &ops);
Expression apply_wick(Expression e);


#endif  // CWICK_SRC_WICK_H_
