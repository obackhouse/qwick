/*
 *  Indices
 */

#ifndef INDEX_H
#define INDEX_H

#include <string>
#include <vector>
#include <algorithm>
#include <variant>

using int_or_string = std::variant<int, std::string>;

class Idx{
    public:
        int index;
        std::string space;
        bool fermion;

        Idx();
        Idx(int _index, std::string _space, bool _fermion=true);
        Idx(std::string _index, std::string _space, bool _fermion=true);

        std::string repr();
};

bool operator==(const Idx &a, const Idx &b);
bool operator!=(const Idx &a, const Idx &b);
bool operator<(const Idx &a, const Idx &b);
bool operator<=(const Idx &a, const Idx &b);
bool operator>(const Idx &a, const Idx &b);
bool operator>=(const Idx &a, const Idx &b);

Idx idx_copy(const Idx &a);

bool is_occupied(const Idx &a, const std::vector<std::string> occ = std::vector<std::string>());

template <class T>
inline void hash_combine(std::size_t& seed, T const& v){
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct IdxHash {
    std::size_t operator()(Idx i) const {
        return std::hash<std::string>()(i.repr());
    }
};

struct IdxStringPairHash {
    std::size_t operator()(std::pair<std::string, Idx> p) const {
        std::size_t seed = std::hash<std::string>()(p.second.repr());
        hash_combine(seed, p.first);
        return seed;
    }
};

#endif
