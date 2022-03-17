/*
 *  Wick theorem
 */

#ifndef WICK_H
#define WICK_H

#include "operator.h"
#include "expression.h"

bool valid_contraction(Operator o1, Operator o2);
std::vector<std::vector<std::pair<Operator, Operator>>> pair_list(std::vector<Operator> lst);
std::pair<int, int> find_pair(int i, std::vector<std::pair<int, int>> ipairs);
int get_sign(std::vector<std::pair<int, int>> ipairs);
std::vector<std::vector<Operator>> split_operators(std::vector<Operator> ops);
Expression apply_wick(Expression e);


#endif
