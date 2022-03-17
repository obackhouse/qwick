/*
 *  Indices
 */

#include<string>
#include<vector>
#include<algorithm>

#include "index.h"

Idx::Idx() {}

Idx::Idx(int _index, std::string _space, bool _fermion) {
    index = _index;
    space = _space;
    fermion = _fermion;
}

std::string Idx::repr() {
    std::string open = "(";
    std::string close = ")";
    std::string out = std::to_string(index) + open + space + close;
    return out;
}

bool operator==(const Idx &a, const Idx &b) {
    return ((a.index == b.index) && (a.space == b.space));
}

bool operator!=(const Idx &a, const Idx &b) {
    return !(a == b);
}

bool operator<(const Idx &a, const Idx &b) {
    bool out;

    if (a.space < b.space) {
        out = true;
    } else if (a.space == b.space) {
        out = a.index < b.index;
    } else {
        out = false;
    }

    return out;
}

bool operator<=(const Idx &a, const Idx &b) {
    return ((a < b) || (a == b));
}

bool operator>(const Idx &a, const Idx &b) {
    return !(a <= b);
}

bool operator>=(const Idx &a, const Idx &b) {
    return !(a < b);
}

Idx idx_copy(const Idx &a) {
    Idx b(a.index, a.space, a.fermion);
    return b;
}

bool is_occupied(const Idx &a, const std::vector<std::string> occ) {
    bool out = false;

    if (occ.size() == 0) {
        out = (a.space.find("o") != std::string::npos);
    }
    else {
        out = (std::find(occ.begin(), occ.end(), a.space) != occ.end());
    }

    return out;
}