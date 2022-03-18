/*
 *  Indices
 */

#ifndef CWICK_SRC_INDEX_H_
#define CWICK_SRC_INDEX_H_

#include <string>
#include <vector>
#include <utility>
#include <variant>
#include <algorithm>


class Idx {
 public:
    // Attributes
    int index;
    std::string space;
    bool fermion;

    // Constructors
    Idx();
    Idx(const int index,
        const std::string space,
        const bool fermion = true);

    // String formatting
    std::string repr() const;
};

bool operator==(const Idx &a, const Idx &b);
bool operator!=(const Idx &a, const Idx &b);
bool operator<(const Idx &a, const Idx &b);
bool operator<=(const Idx &a, const Idx &b);
bool operator>(const Idx &a, const Idx &b);
bool operator>=(const Idx &a, const Idx &b);

Idx idx_copy(const Idx &a);

bool is_occupied(const Idx &a, const std::vector<std::string> &occ = std::vector<std::string>());

struct IdxHash {
    std::size_t operator()(const Idx i) const {
        return std::hash<std::string>()(i.repr());
    }
};

struct IdxStringPairHash {
    std::size_t operator()(const std::pair<std::string, Idx> p) const {
        std::size_t seed = std::hash<std::string>()(p.second.repr());
        seed ^= std::hash<std::string>()(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

#endif  // CWICK_SRC_INDEX_H_
