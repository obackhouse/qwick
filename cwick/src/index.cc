/*
 *  Indices
 */

#include "util.h"
#include "index.h"


Idx::Idx() {}

Idx::Idx(const int _index,
         const std::string _space,
         const bool _fermion) {
    index = _index;
    space = _space;
    fermion = _fermion;
}

std::string Idx::repr() const {
    return std::to_string(index) + "(" + space + ")";
}

bool operator==(const Idx &a, const Idx &b) {
    return ((a.index == b.index) && (a.space == b.space));
}

bool operator!=(const Idx &a, const Idx &b) {
    return (!(a == b));
}

bool operator<(const Idx &a, const Idx &b) {
    if (a.space < b.space) {
        return true;
    } else if (a.space == b.space) {
        return a.index < b.index;
    } else {
        return false;
    }
}

bool operator<=(const Idx &a, const Idx &b) {
    return ((a < b) || (a == b));
}

bool operator>(const Idx &a, const Idx &b) {
    return (!(a <= b));
}

bool operator>=(const Idx &a, const Idx &b) {
    return (!(a < b));
}

Idx idx_copy(const Idx &a) {
    return Idx(a.index, a.space, a.fermion);
}

bool is_occupied(const Idx &a, const std::vector<std::string> &occ) {
    if (occ.size() == 0) {
        return (a.space.find("o") != std::string::npos);
    } else {
        return (std::find(occ.begin(), occ.end(), a.space) != occ.end());
    }
}
