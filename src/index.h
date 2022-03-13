/*
 *  Indices
 */

#ifndef INDEX_H
#define INDEX_H

#include <string>

class Idx{
    public:
        int index;
        std::string space;
        bool fermion;

        void init(int index, std::string space, bool fermion);
        std::string repr();
};

bool operator==(const Idx &a, const Idx &b);
bool operator!=(const Idx &a, const Idx &b);
bool operator<(const Idx &a, const Idx &b);
bool operator<=(const Idx &a, const Idx &b);
bool operator>(const Idx &a, const Idx &b);
bool operator>=(const Idx &a, const Idx &b);

Idx idx_copy(const Idx &a);

bool is_occupied(const Idx &a);

template <class T>
inline void hash_combine(std::size_t& seed, T const& v){
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct IdxHash {
    std::size_t operator()(const Idx i) const {
        std::size_t seed = 0;
        hash_combine(seed, i.index);
        hash_combine(seed, i.space);
        hash_combine(seed, i.fermion);
        return seed;
    }
};

struct IdxStringPairHash {
    std::size_t operator()(const std::pair<std::string, Idx> p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second.index);
        hash_combine(seed, p.second.space);
        hash_combine(seed, p.second.fermion);
        return seed;
    }
};

#endif
