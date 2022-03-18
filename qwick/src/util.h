/*
 *  C++ specific utilities
 */

#ifndef QWICK_SRC_UTIL_H_
#define QWICK_SRC_UTIL_H_

#define TRAILING_ZERO_PRECISION 12

#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <numeric>
#include <utility>
#include <algorithm>

std::string format_float(const double val, const bool trailing_zero = false);

template<typename T>
std::vector<std::size_t> argsort(const std::vector<T> &vec) {
    // https://stackoverflow.com/questions/17074324
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), 
            [&](std::size_t i, std::size_t j){ return vec[i] < vec[j]; });

    return p;
}

template<typename T>
std::vector<T> apply_argsort(const std::vector<T> &vec, const std::vector<std::size_t> &p) {
    // https://stackoverflow.com/questions/17074324
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });

    return sorted_vec;
}

#endif  // QWICK_SRC_UTIL_H_
