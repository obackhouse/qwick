/*
 *  Expressions
 */

#include "index.h"
#include "operator.h"
#include "expression.h"

#define swap(x, y) { auto temp = x; x = y; y = temp; }

/*
 *  Wick implements TermMaps as sets of complex nested data structures
 *  which are pretty horrible to implement in C++. Instead, TermMap can
 *  be a vector which can be sorted in O(N log N) and then equality can
 *  be checked in O(N) worst-case. TermMap instances don't have very
 *  long lengths in any practical example so this might be fine.
 *
 *  TermMap is only used to compare equality between two instances.
 */

void TermMap::init(std::vector<Sigma> sums, std::vector<Tensor> tensors) {
    for (unsigned int _ti = 0; _ti < tensors.size(); _ti++) {
        Tensor ti = tensors[_ti];  // TODO make reference
        std::vector<Idx> ilist;
        ilist = ti.ilist();

        std::string tiname;
        if (ti.name == "") {
            tiname = "!";
        } else {
            tiname = ti.name;
        }

        // can't be arsed with a dict, use two vectors instead FIXME
        std::vector<std::string> strs_keys;
        std::vector<std::unordered_set<std::string>> strs_vals;
        for (unsigned int x = 0; x < ilist.size(); x++) {
            strs_keys.push_back(ilist[x].space);
        }

        for (unsigned int i = 0; i < ti.indices.size(); i++) {
            Idx iidx = ti.indices[i];  // TODO make reference
            std::string space = iidx.space;
            std::string istr = std::to_string(i);

            bool summed = false;
            for (unsigned int x = 0; x < sums.size(); x++) {
                if (sums[x].idx == iidx) {
                    summed = true;
                    break;
                }
            }

            for (unsigned int _tj = 0; _tj < tensors.size(); _tj++) {
                Tensor tj = tensors[_tj];  // TODO make reference

                std::string tjname;
                if (tj.name == "") {
                    tjname = "!";
                } else {
                    tjname = tj.name;
                }

                // jts
                std::unordered_set<std::pair<std::string, Idx>, IdxStringPairHash> jts;
                for (unsigned int j = 0; j < tj.indices.size(); j++) {
                    Idx jidx = tj.indices[j];  // TODO make reference
                    std::pair<std::string, Idx> p;
                    p.first = std::to_string(j);
                    p.second = jidx;
                    jts.insert(p);
                }

                // if jts, add to strs "dict"
                if (jts.size() != 0) {
                    // find index in strs_keys that matches space
                    int strs_idx = -1;
                    for (unsigned int x = 0; x < strs_keys.size(); x++) {
                        if (strs_keys[x] == space) {
                            strs_idx = x;
                            break;
                        }
                    }
                    assert(strs_idx != -1);

                    //std::unordered_set<std::pair<std::string, Idx>, IdxStringPairHash>::iterator itr;
                    for (auto itr = jts.begin(); itr != jts.end(); ++itr) {
                        // make_str
                        std::string x_str = istr + tjname + (*itr).first;
                        if (summed) {
                            x_str = x_str + "x";
                        }
                        strs_vals[strs_idx].insert(x_str);
                    }
                }
            }
        }

        // lll: sort strs by the keys
        // strs: strs_keys (vector of str) : strs_vals (vector of set<str>)
        // sort them in-place with a brute-force algo for now:  FIXME
        {
            unsigned int x, y, min;
            for (x = 0; x < strs_vals.size()-1; x++) {
                for (y = x; y < strs_vals.size(); y++) {
                    if (strs_keys[y] < strs_keys[min]) {
                        min = y;
                    }
                    swap(strs_keys[x], strs_keys[min]);
                    swap(strs_vals[x], strs_vals[min]);
                }
            }
        }

        // make element of TermMap.data
        TERM_MAP_DATA_TYPE elem;
        elem.first = tiname;
        for (unsigned int x = 0; x < strs_keys.size(); x++) {
            std::pair<std::string, std::unordered_set<std::string>> p;
            p.first = strs_keys[x];
            p.second = strs_vals[x];

            elem.second.push_back(p);
        }

        // append element if not in already - may scale too badly FIXME
        bool found = false;
        for (unsigned int x = 0; x < data.size(); x++) {
            if (elem == data[x]) {
                found = true;
                break;
            }
        }
        if (!(found)) {
            data.push_back(elem);
        }
    }

    // kill me
}

// FIXME single breakpoint plz
bool operator==(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b) {
    if (a.first != b.first) {
        return false;
    } else if (a.second.size() != b.second.size()) {
        return false;
    } else {
        for (unsigned int i = 0; i < a.second.size(); i++) {
            if (a.second[i].first != b.second[i].first) {
                return false;
            } else if (a.second[i].second != b.second[i].second) {
                return false;
            }
        }
    }

    return true;
}

bool operator!=(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b) {
    return (!(a == b));
}

// FIXME when vector is changed to set
bool operator==(const TermMap &a, const TermMap &b) {
    if (a.data.size() != b.data.size()) {
        return false;
    } else {
        for (unsigned int i = 0; i < a.data.size(); i++) {
            if (a.data[i] != b.data[i]) {
                return false;
            }
        }
    }
    
    return true;
}

bool operator!=(const TermMap &a, const TermMap &b) {
    return (!(a == b));
}


int get_case(Delta dd, std::unordered_set<Idx, IdxHash> ilist) {
    assert(dd.i1.space == dd.i2.space);

    std::unordered_set<Idx, IdxHash>::iterator it1 = ilist.find(dd.i1);
    bool is1 = (it1 != ilist.end());

    std::unordered_set<Idx, IdxHash>::iterator it2 = ilist.find(dd.i2);
    bool is2 = (it2 != ilist.end());

    int _case = 0;
    if (is1) {
        _case = 1;
    }
    if (is2) {
        if (_case == 0) {
            _case = 2;
        } else {
            _case = 3;
        }
    }

    return _case;
}

void _resolve(
        std::vector<Sigma> &sums,
        std::vector<Tensor> &tensors,
        std::vector<Operator> &operators,
        std::vector<Delta> &deltas)
{
    std::vector<Sigma> newsums;
    std::vector<Tensor> newtensors;
    std::vector<Operator> newoperators;
    std::vector<Delta> newdeltas;

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums.push_back(sums[i].copy());
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors.push_back(tensors[i].copy());
    }
    for (unsigned int i = 0; i < operators.size(); i++) {
        newoperators.push_back(operators[i].copy());
    }

    for (unsigned int i = 0; i < deltas.size(); i++) {
        for (unsigned int j = 0; j < newdeltas.size(); j++) {
            if (deltas[i] == newdeltas[j]) {
                break;
            }
        }
        newdeltas.push_back(deltas[i].copy());
    }

    std::unordered_set<Idx, IdxHash> ilist;
    for (unsigned int i = 0; i < sums.size(); i++) {
        ilist.insert(sums[i].idx);
    }

    // TODO just inline this
    std::vector<int> cases;
    for (unsigned int d = 0; d < newdeltas.size(); d++) {
        cases[d] = get_case(newdeltas[d], ilist);
    }

    std::vector<Delta> rs;

    // Iterate over a copy of cases
    std::vector<int> cases1(cases);
    for (unsigned int d = 0; d < newdeltas.size(); d++) {
        int _case = cases1[d];
        Idx i1 = newdeltas[d].i1;
        Idx i2 = newdeltas[d].i2;

        if (_case == 3) {
            continue;
        } else if (_case == 1) {
            Sigma sigma_i1;
            sigma_i1.init(i1);
            bool found = false;
            for (unsigned int x = 0; x < newsums.size(); x++) {
                if (newsums[x] == sigma_i1) {
                    newsums.erase(newsums.begin() + x);
                    found = true;
                    break;
                }
            }
            assert(found);
        } else if (_case == 2) {
            Sigma sigma_i2;
            sigma_i2.init(i2);
            bool found = false;
            for (unsigned int x = 0; x < newsums.size(); x++) {
                if (newsums[x] == sigma_i2) {
                    newsums.erase(newsums.begin() + x);
                    found = true;
                    break;
                }
            }
            assert(found);
        } else {
            assert(_case == 0);
        }

        // Iterate over a copy of cases
        std::vector<int> cases2(cases);
        for (unsigned int i = 0; i < newdeltas.size(); i++) {
            int ccc = cases2[i];

            if ((_case == 1) && (newdeltas[i].i1 == i1)) {
                newdeltas[i].i1 = i2;
                if (ccc == 3) {
                    cases[i] = 2;
                } else if (ccc == 1) {
                    cases[i] = 0;
                } else {
                    assert(0);
                }
            } else if ((_case == 1) && (newdeltas[i].i2 == i1)) {
                newdeltas[i].i2 = i2;
                if (ccc == 3) {
                    cases[i] = 1;
                } else if (ccc == 2) {
                    cases[i] = 0;
                } else {
                    assert(0);
                }
            } else if ((_case == 2) && (newdeltas[i].i2 == i2)) {
                newdeltas[i].i2 = i1;
                if (ccc == 3) {
                    cases[i] = 1;
                } else if (ccc == 2) {
                    cases[i] = 0;
                } else {
                    assert(0);
                }
            } else if ((_case == 2) && (newdeltas[i].i1 == i2)) {
                newdeltas[i].i1 = i1;
                if (ccc == 3) {
                    cases[i] = 2;
                } else if (ccc == 1) {
                    cases[i] = 0;
                } else {
                    assert(0);
                }
            }
        }

        for (unsigned int t = 0; t < newtensors.size(); t++) {
            for (unsigned int k = 0; k < newtensors[t].indices.size(); k++) {
                if (_case == 1) {
                    if (newtensors[t].indices[k] == i1) {
                        newtensors[t].indices[k] = i2;
                    }
                } else if (_case == 2) {
                    if (newtensors[t].indices[k] == i2) {
                        newtensors[t].indices[k] = i1;
                    }
                }
            }
        }

        for (unsigned int o = 0; o < newoperators.size(); o++) {
            if (_case == 1) {
                if (newoperators[o].idx == i1) {
                    newoperators[o].idx = i2;
                }
            } else if (_case == 2) {
                if (newoperators[o].idx == i2) {
                    newoperators[o].idx = i1;
                }
            }
        }

        if (!((_case == 0) && (i1 != i2))) {
            rs.push_back(deltas[d]);
        }
    }

    for (unsigned int i = 0; i < rs.size(); i++) {
        bool found = false;
        for (unsigned int x = 0; x < newdeltas.size(); x++) {
            if (newdeltas[x] == rs[i]) {
                newdeltas.erase(newdeltas.begin() + x);
                cases.erase(cases.begin() + x);
                found = true;
                break;
            }
        }
        assert(found);
    }

    for (unsigned int i = 0; i < cases.size(); i++) {
        if ((cases[i] == 1) || (cases[i] == 2)) {
            _resolve(newsums, newtensors, newoperators, newdeltas);
            return;
        }
    }

    assert(newdeltas.size() == cases.size());
    for (unsigned int d = 0; d < newdeltas.size(); d++) {
        int _case = cases[d];
        Idx i1 = newdeltas[d].i1;
        Idx i2 = newdeltas[d].i2;

        if (_case == 3) {
            Sigma sigma_i2;
            sigma_i2.init(i2);
            bool found = false;
            for (unsigned int x = 0; x < newsums.size(); x++) {
                if (newsums[x] == sigma_i2) {
                    newsums.erase(newsums.begin() + x);
                    found = true;
                    break;
                }
            }
            assert(found);
        } else if (_case < 3) {
            assert(_case == 0);
        } else {
            assert(0);
        }

        if (_case == 0) {
            continue;
        }

        for (unsigned int t = 0; t < newtensors.size(); t++) {
            for (unsigned int k = 0; k < newtensors[t].indices.size(); k++) {
                if (newtensors[t].indices[k] == i2) {
                    newtensors[t].indices[k] = i1;
                }
            }
        }

        for (unsigned int o = 0; o < newoperators.size(); o++) {
            if (newoperators[o].idx == i2) {
                newoperators[o].idx = i1;
            }
        }

        rs.push_back(newdeltas[d]);
    }


    for (unsigned int i = 0; i < rs.size(); i++) {
        bool found = false;
        for (unsigned int x = 0; x < newdeltas.size(); x++) {
            if (newdeltas[x] == rs[i]) {
                newdeltas.erase(newdeltas.begin() + x);
                found = true;
                break;
            }
        }
        assert(found);
    }

    // Set input arrays to the new arrays
    sums = newsums;
    tensors = newtensors;
    operators = newoperators;
    deltas = newdeltas;

    return;
}


std::unordered_map<std::string, std::string> default_index_key() {
    std::unordered_map<std::string, std::string> out = {
        {"occ", "ijklmnop"},
        {"vir", "abcdefgh"},
        {"nm", "IJKLMNOP"},
    };
    return out;
};

void Term::init(
        double _scalar,
        std::vector<Sigma> _sums,
        std::vector<Tensor> _tensors,
        std::vector<Operator> _operators,
        std::vector<Delta> _deltas,
        std::unordered_map<std::string, std::string> _index_key=default_index_key())
{
    scalar = _scalar;
    sums = _sums;
    tensors = _tensors;
    operators = _operators;
    deltas = _deltas;
    index_key = _index_key;
}

// TODO should Term.resolve return another Term or change in-place?
void Term::resolve() {
    _resolve(sums, tensors, operators, deltas);
}

std::string Term::repr() {
    std::string out = "";

    for (unsigned int i = 0; i < sums.size(); i++) {
        out = out + sums[i].repr();
    }
    for (unsigned int i = 0; i < deltas.size(); i++) {
        out = out + deltas[i].repr();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        out = out + tensors[i].repr();
    }
    for (unsigned int i = 0; i < operators.size(); i++) {
        out = out + operators[i].repr();
    }

    return out;
}

std::vector<Idx> Term::ilist() {
    std::unordered_set<Idx, IdxHash> sout;
    std::vector<Idx> out;

    for (unsigned int i = 0; i < operators.size(); i++) {
        sout.insert(operators[i].idx);
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        std::vector<Idx> ilist_tensor = tensors[i].ilist();
        for (unsigned int j = 0; j < ilist_tensor.size(); j++) {
            sout.insert(ilist_tensor[j]);
        }
    }
    for (unsigned int i = 0; i < sums.size(); i++) {
        sout.insert(sums[i].idx);
    }
    for (unsigned int i = 0; i < deltas.size(); i++) {
        sout.insert(deltas[i].i1);
        sout.insert(deltas[i].i2);
    }

    for (auto itr = sout.begin(); itr != sout.end(); ++itr) {
        out.push_back(*itr);
    }

    // sort FIXME
    {
        unsigned int x, y, min;
        for (x = 0; x < out.size()-1; x++) {
            for (y = x; y < out.size(); y++) {
                if (out[y] < out[min]) {
                    min = y;
                }
                swap(out[x], out[min]);
            }
        }
    }

    return out;
}

Term Term::_inc(int i) {
    std::vector<Sigma> newsums;
    std::vector<Tensor> newtensors;
    std::vector<Operator> newoperators;
    std::vector<Delta> newdeltas;

    for (unsigned int j = 0; j < sums.size(); j++) {
        newsums.push_back(sums[j]._inc(i));
    }
    for (unsigned int j = 0; j < tensors.size(); j++) {
        newtensors.push_back(tensors[j]._inc(i));
    }
    for (unsigned int j = 0; j < operators.size(); j++) {
        newoperators.push_back(operators[j]._inc(i));
    }
    for (unsigned int j = 0; j < deltas.size(); j++) {
        newdeltas.push_back(deltas[j]._inc(i));
    }

    Term out;
    out.init(scalar, newsums, newtensors, newoperators, newdeltas, index_key);

    return out;
}

Term Term::copy() {
    double newscalar = scalar;
    std::vector<Sigma> newsums;
    std::vector<Tensor> newtensors;
    std::vector<Operator> newoperators;
    std::vector<Delta> newdeltas;

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums.push_back(sums[i].copy());
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors.push_back(tensors[i].copy());
    }
    for (unsigned int i = 0; i < operators.size(); i++) {
        newoperators.push_back(operators[i].copy());
    }
    for (unsigned int i = 0; i < deltas.size(); i++) {
        newdeltas.push_back(deltas[i].copy());
    }

    Term out;
    out.init(newscalar, newsums, newtensors, newoperators, newdeltas, index_key);

    return out;
}

Term operator*(Term &a, Term &b) {
    std::vector<Idx> il1 = a.ilist();
    std::vector<Idx> il2 = b.ilist();
    std::unordered_set<Idx, IdxHash> sil1, sil2, sil12_intersection;

    for (unsigned int i = 0; i < il1.size(); i++) {
        sil1.insert(il1[i]);
    }
    for (unsigned int i = 0; i < il2.size(); i++) {
        sil2.insert(il2[i]);
    }
    for (
        std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
        itr != sil2.end();
        itr++)
    {
        if (sil2.find(*itr) != sil2.end()) {
            sil12_intersection.insert(*itr);
        }
    }

    Term newb;
    if (sil12_intersection.size() != 0) {
        std::vector<int> sil1_indices;
        for (
            std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
            itr != sil1.end();
            itr++)
        {
            sil1_indices.push_back((*itr).index);
        }
        int m = *max_element(sil1_indices.begin(), sil1_indices.begin());
        newb = b._inc(m + 1);
    } else {
        newb = b;  // TODO reference
    }

    double newscalar = a.scalar * newb.scalar;

    std::vector<Sigma> newsums = a.sums;
    std::vector<Tensor> newtensors = a.tensors;
    std::vector<Operator> newoperators = a.operators;
    std::vector<Delta> newdeltas = a.deltas;

    newsums.insert(newsums.end(), newb.sums.begin(), newb.sums.end());
    newtensors.insert(newtensors.end(), newb.tensors.begin(), newb.tensors.end());
    newoperators.insert(newoperators.end(), newb.operators.begin(), newb.operators.end());
    newdeltas.insert(newdeltas.end(), newb.deltas.begin(), newb.deltas.end());

    std::unordered_map<std::string, std::string> new_index_key;

    if (a.index_key == default_index_key()) {
        new_index_key = b.index_key;
    } else {
        new_index_key = a.index_key;
    }

    Term newterm;
    newterm.init(newscalar, newsums, newtensors, newoperators, newdeltas, new_index_key);

    return newterm;
}

Term operator*(Term &a, double &b) {
    Term newterm = a.copy();
    newterm.scalar *= b;
    return newterm;
}

Term operator*(double &a, Term &b) {
    return (b * a);
}

Term operator*(Term &a, int &b) {
    Term newterm = a.copy();
    newterm.scalar = newterm.scalar * b;
    return newterm;
}

Term operator*(int &a, Term &b) {
    return (b * a);
}

// FIXME single breakpoint
// FIXME these comparisons probably need to be cast to set for O(1)
bool operator==(const Term &a, const Term &b) {
    if (a.scalar != b.scalar) {
        return false;
    } else if (!(std::is_permutation(a.sums.begin(), a.sums.end(), b.sums.begin()))) {
        return false;
    } else if (!(std::is_permutation(a.tensors.begin(), a.tensors.end(), b.tensors.begin()))) {
        return false;
    } else if (!(std::is_permutation(a.deltas.begin(), a.deltas.end(), b.deltas.begin()))) {
        return false;
    } else {
        // This one is ordered!
        if (a.operators.size() != b.operators.size()) {
            return false;
        } else {
            for (unsigned int i = 0; i < a.operators.size(); i++) {
                if (a.operators[i] != b.operators[i]) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool operator!=(const Term &a, const Term &b) {
    return (!(a == b));
}

void ATerm::init(
            double _scalar,
            std::vector<Sigma> _sums,
            std::vector<Tensor> _tensors,
            std::unordered_map<std::string, std::string> _index_key=default_index_key())
{
    scalar = _scalar;
    sums = _sums;
    tensors = _tensors;
    index_key = _index_key;
}

void ATerm::init(Term term) {
    scalar = term.scalar;
    index_key = term.index_key;

    assert(term.operators.size() == 0);

    for (unsigned int i = 0; i < term.sums.size(); i++) {
        sums.push_back(term.sums[i].copy());
    }
    for (unsigned int i = 0; i < term.tensors.size(); i++) {
        tensors.push_back(term.tensors[i].copy());
    }
    for (unsigned int i = 0; i < term.deltas.size(); i++) {
        tensors.push_back(tensor_from_delta(term.deltas[i]));
    }
}

std::string ATerm::repr() {
    std::string out = "";

    for (unsigned int i = 0; i < sums.size(); i++) {
        out = out + sums[i].repr();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        out = out + tensors[i].repr();
    }

    return out;
}

ATerm ATerm::_inc(int i) {
    std::vector<Sigma> newsums;
    std::vector<Tensor> newtensors;

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums.push_back(sums[i]._inc(i));
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors.push_back(tensors[i]._inc(i));
    }

    ATerm out;
    out.init(scalar, newsums, newtensors, index_key);

    return out;
}

bool ATerm::match(ATerm other) {
    TermMap TM1, TM2;
    TM1.init(sums, tensors);
    TM2.init(other.sums, other.tensors);
    return TM1 == TM2;
}

// Returns 0 instead of None
int ATerm::pmatch(ATerm other) {
    if (sums.size() != other.sums.size()) {
        return 0;
    }
    if (tensors.size() != other.tensors.size()) {
        return 0;
    }

    std::vector<std::vector<std::pair<std::vector<int>, int>>> tlists;
    for (unsigned int i = 0; i < other.tensors.size(); i++) {
        tlists.push_back(other.tensors[i].sym.tlist);
    }

    TermMap TM1;
    TM1.init(sums, tensors);

    // Loop over a cartesian product of elements of tlists
    // https://stackoverflow.com/questions/5279051
    auto product = [](
        long long a,
        std::vector<std::pair<std::vector<int>, int>>& b)
    {
        return a*b.size();
    };
    const long long N = std::accumulate(tlists.begin(), tlists.end(), 1LL, product);
    std::vector<std::pair<std::vector<int>, int>> u(tlists.size());

    for (long long n = 0; n < N; n++) {
        std::lldiv_t q {n, 0};
        for (long long i = tlists.size()-1; i >= 0; i--) {
            q = std::div(q.quot, tlists[i].size());
            u[i] = tlists[i][q.rem];
        }

        int sign = 1;
        for (unsigned int x = 0; x < u.size(); x++) {
            sign *= u[x].second;
        }

        // permute
        assert(other.tensors.size() == u.size());
        std::vector<Tensor> newtensors;
        for (unsigned int x = 0; x < other.tensors.size(); x++) {
            newtensors.push_back(permute(other.tensors[x], u[x].first));
        }

        TermMap TM2;
        TM2.init(other.sums, newtensors);
                   
        if (TM1 == TM2) {
            return sign;
        }
    }

    return 0;
}

// FIXME scales badly
std::vector<Idx> ATerm::ilist() {
    std::vector<Idx> out;

    for (unsigned int i = 0; i < tensors.size(); i++) {
        std::vector<Idx> ilist_tensor = tensors[i].ilist();
        for (unsigned int j = 0; j < ilist_tensor.size(); j++) {
            bool found = false;
            for (unsigned int k = 0; k < out.size(); k++) {
                if (ilist_tensor[j] == out[k]) {
                    found = true;
                    break;
                }
            }
            if (!(found)) {
                out.push_back(ilist_tensor[j]);
            }
        }
    }

    for (unsigned int i = 0; i < sums.size(); i++) {
        bool found = false;
        for (unsigned int j = 0; j < out.size(); j++) {
            if (sums[i].idx == out[j]) {
                found = true;
                break;
            }
        }
        if (!(found)) {
            out.push_back(sums[i].idx);
        }
    }

    return out;
}

unsigned int ATerm::nidx() {
    std::vector<Idx> idxs = ilist();
    return idxs.size();
}

void ATerm::sort_tensors() {
    unsigned int off = 0;
    for (unsigned int i = 0; i < tensors.size(); i++) {
        if (tensors[i].name == "") {
            swap(tensors[i], tensors[off]);
            off += 1;
        }
    }
}

void ATerm::merge_external() {
    bool ext = true;

    for (unsigned int t = 0; t > tensors.size(); t++) {
        if ((!(ext)) && (tensors[t].name == "")) {
            assert(0);
        }
        if (tensors[t].name == "") {
            ext = false;
        }
    }

    unsigned int num_ext = 0;
    for (unsigned int t = 0; t < tensors.size(); t++) {
        if (tensors[t].name == "") {
            num_ext++;
        }
    }

    if (num_ext > 1) {
        std::vector<Tensor> newtensors;
        for (unsigned int t = num_ext; t < tensors.size(); t++) {
            newtensors.push_back(tensors[t].copy());
        }

        std::vector<Idx> ext_indices;
        for (unsigned int t = num_ext; t < tensors.size(); t++) {
            for (unsigned int i = 0; i < tensors[t].indices.size(); i++) {
                ext_indices.push_back(tensors[t].indices[i]);
            }
        }

        // TODO remove if default value is set
        TensorSym sym;
        std::vector<int> plist;
        for (unsigned int i = 0; i < ext_indices.size(); i++) {
            plist.push_back(0);
        }
        sym.init({plist}, {0});

        Tensor t_ext;
        t_ext.init(ext_indices, "", sym);

        tensors.insert(tensors.begin(), t_ext);
    }
}

bool ATerm::connected() {
    std::vector<Idx> ll;
    for (unsigned int i = 0; i < sums.size(); i++) {
        ll.push_back(sums[i].idx);
    }

    std::vector<Tensor> rtensors;
    for (unsigned int i = 0; i < tensors.size(); i++) {
        if ((tensors[i].name != "") && (tensors[i].indices.size() != 0)) {
            rtensors.push_back(tensors[i]);
        }
    }

    std::vector<std::unordered_set<int>> adj;
    for (unsigned int i = 0; i < ll.size(); i++) {
        for (unsigned int j = 0; j < rtensors.size(); j++) {
            bool found = false;
            for (unsigned int k = 0; k < rtensors[j].indices.size(); k++) {
                if (ll[i] == rtensors[j].indices[k]) {
                    found = true;
                }
            }
            if (found) {
                adj[i].insert(j);
            }
        }
    }

    if (adj.size() == 0) {
        return (rtensors.size() < 2);
    }

    std::unordered_set<int> blue = adj[0];
    int nb = blue.size();
    int maxiter = 300000;
    for (int i = 0; i < maxiter; i++) {
        std::vector<int> newtensors;

        for (
            std::unordered_set<int>::iterator b = blue.begin();
            b != blue.end();
            b++)
        {
            for (unsigned int a = 0; a < adj.size(); a++) {
                if ((adj[a]).find(*b) != (adj[a]).end()) {
                    for (
                        std::unordered_set<int>::iterator itr = (adj[a]).begin();
                        itr != (adj[a]).end();
                        itr++)
                    {
                        newtensors.push_back(*itr);
                    }
                }
            }
        }

        for (unsigned int j = 0; j < newtensors.size(); j++) {
            blue.insert(newtensors[j]);
        }
        int nb2 = blue.size();
        if (nb2 == nb) {
            break;
        }
        nb = nb2;
    }

    return (blue.size() == rtensors.size());
}

bool ATerm::reducible() {
    if (!(connected())) {
        return true;
    }

    for (unsigned int i = 0; i < sums.size(); i++) {
        ATerm newterm = copy();
        // FIXME there is a newterm._inc(1) line here in wick, but it does nothing.
        Idx i1 = newterm.sums[i].idx;

        std::vector<Sigma> newsums;
        for (unsigned int j = 0; j < sums.size(); j++) {
            if (sums[j] != newterm.sums[i]) {
                newsums.push_back(sums[j]);
            }
        }
        newterm.sums = newsums;

        int m = 0;
        for (unsigned int t = 0; t < newterm.tensors.size(); t++) {
            for (unsigned int x = 0; x < newterm.tensors[t].indices.size(); x++) {
                if (newterm.tensors[t].indices[x] == i1) {
                    m += 1;
                }
            }
        }

        assert(m == 2);

        if (!(newterm.connected())) {
            return true;
        }
    }

    return false;
}

void ATerm::transpose(std::vector<int> perm) {
    merge_external();
    tensors[0].transpose(perm);
}

ATerm ATerm::copy() {
    std::vector<Tensor> newtensors;
    std::vector<Sigma> newsums;

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums.push_back(sums[i].copy());
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors.push_back(tensors[i].copy());
    }

    ATerm out;
    out.init(scalar, newsums, newtensors, index_key);

    return out;
}

ATerm operator*(ATerm &a, ATerm &b) {
    std::vector<Idx> il1 = a.ilist();
    std::vector<Idx> il2 = b.ilist();
    std::unordered_set<Idx, IdxHash> sil1, sil2, sil12_intersection;

    for (unsigned int i = 0; i < il1.size(); i++) {
        sil1.insert(il1[i]);
    }
    for (unsigned int i = 0; i < il2.size(); i++) {
        sil2.insert(il2[i]);
    }
    for (
        std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
        itr != sil2.end();
        itr++)
    {
        if (sil2.find(*itr) != sil2.end()) {
            sil12_intersection.insert(*itr);
        }
    }

    ATerm newb;
    if (sil12_intersection.size() != 0) {
        std::vector<int> sil1_indices;
        for (
            std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
            itr != sil1.end();
            itr++)
        {
            sil1_indices.push_back((*itr).index);
        }
        int m = *max_element(sil1_indices.begin(), sil1_indices.begin());
        newb = b._inc(m + 1);
    } else {
        newb = b;  // TODO reference
    }

    double newscalar = a.scalar * newb.scalar;

    std::vector<Sigma> newsums = a.sums;
    std::vector<Tensor> newtensors = a.tensors;

    newsums.insert(newsums.end(), newb.sums.begin(), newb.sums.end());
    newtensors.insert(newtensors.end(), newb.tensors.begin(), newb.tensors.end());

    std::unordered_map<std::string, std::string> new_index_key;

    if (a.index_key == default_index_key()) {
        new_index_key = b.index_key;
    } else {
        new_index_key = a.index_key;
    }

    ATerm newterm;
    newterm.init(newscalar, newsums, newtensors, new_index_key);

    return newterm;
}

ATerm operator*(ATerm &a, double &b) {
    ATerm newterm = a.copy();
    newterm.scalar *= b;
    return newterm;
}

ATerm operator*(double &a, ATerm &b) {
    return (b * a);
}

ATerm operator*(ATerm &a, int &b) {
    ATerm newterm = a.copy();
    newterm.scalar = newterm.scalar * b;
    return newterm;
}

ATerm operator*(int &a, ATerm &b) {
    return (b * a);
}

// FIXME these comparisons probably need to be cast to set for O(1)
bool operator==(const ATerm &a, const ATerm &b) {
    if (a.scalar != b.scalar) {
        return false;
    } else if (!(std::is_permutation(a.sums.begin(), a.sums.end(), b.sums.begin()))) {
        return false;
    } else if (!(std::is_permutation(a.tensors.begin(), a.tensors.end(), b.tensors.begin()))) {
        return false;
    }

    return true;
}

bool operator!=(const ATerm &a, const ATerm &b) {
    return (!(a == b));
}

bool operator<(const ATerm &a, const ATerm &b) {
    if (a.tensors.size() < b.tensors.size()) {
        return true;
    } else if (a.tensors.size() == b.tensors.size()) {
        if (a.sums.size() == b.sums.size()) {
            bool same_tensors = true;
            for (unsigned int i = 0; i < a.tensors.size(); i++) {
                if (a.tensors[i] != b.tensors[i]) {
                    same_tensors = false;
                    break;
                }
            }
            if (same_tensors) {
                for (unsigned int i = 0; i < a.sums.size(); i++) {
                    if (a.sums[i] != b.sums[i]) {
                        return (a.sums[i] < b.sums[i]);
                    }
                }
            } else {
                for (unsigned int i = 0; i < a.tensors.size(); i++) {
                    if (a.tensors[i] != b.tensors[i]) {
                        return (a.tensors[i] < b.tensors[i]);
                    }
                }
            }
        } else {
            return (a.sums.size() < b.sums.size());
        }
    } else {
        return false;
    }

    return false;
}

bool operator<=(const ATerm &a, const ATerm &b) {
    return ((a < b) || (a == b));
}

bool operator>(const ATerm &a, const ATerm &b) {
    return (!(a <= b));
}

bool operator>=(const ATerm &a, const ATerm &b) {
    return (!(a < b));
}

void Expression::init(std::vector<Term> _terms) {
    terms = _terms;
}

void Expression::resolve() {
    for (unsigned int i = 0; i < terms.size(); i++) {
        terms[i].resolve();
    }

    std::vector<Term> newterms;
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (abs(terms[i].scalar) > tthresh) {
            newterms.push_back(terms[i]);
        }
    }

    terms = newterms;
}

std::string Expression::repr() {
    std::string out = "";
    
    for (unsigned int i = 0; i < terms.size(); i++) {
        out = out + terms[i].repr();
        if (i != (terms.size()-1)) {
            out = out + " + ";
        }
    }

    return out;
}

bool Expression::are_operators() {
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (terms[i].operators.size() > 0) {
            return true;
        }
    }

    return false;
}

Expression operator+(const Expression &a, const Expression &b) {
    std::vector<Term> newterms(a.terms);
    for (unsigned int i = 0; i < b.terms.size(); i++) {
        newterms.push_back(b.terms[i]);
    }

    Expression out;
    out.init(newterms);

    return out;
}

Expression operator-(Expression &a, Expression &b) {
    double fac = -1.0;
    Expression mb = b * fac;
    return (a + mb);
}

Expression operator*(Expression &a, Expression &b) {
    std::vector<Term> newterms;

    for (unsigned int i = 0; i < a.terms.size(); i++) {
        for (unsigned int j = 0; j < b.terms.size(); j++) {
            newterms.push_back(a.terms[i] * b.terms[j]);
        }
    }

    Expression out;
    out.init(newterms);

    return out;
}

Expression operator*(Expression &a, double &b) {
    std::vector<Term> newterms;

    for (unsigned int i = 0; i < a.terms.size(); i++) {
        newterms.push_back(a.terms[i] * b);
    }

    Expression out;
    out.init(newterms);

    return out;
}

Expression operator*(double &a, Expression &b) {
    return (b * a);
}

Expression operator*(Expression &a, int &b) {
    std::vector<Term> newterms;

    for (unsigned int i = 0; i < a.terms.size(); i++) {
        newterms.push_back(a.terms[i] * b);
    }

    Expression out;
    out.init(newterms);

    return out;
}

Expression operator*(int &a, Expression &b) {
    return (b * a);
}

bool operator==(const Expression &a, const Expression &b) {
    if (a.terms.size() != b.terms.size()) {
        return false;
    }

    for (unsigned int i = 0; i < a.terms.size(); i++) {
        if (a.terms[i] != b.terms[i]) {
            return false;
        }
    }

    return true;
}

bool operator!=(const Expression &a, const Expression &b) {
    return (!(a == b));
}


void AExpression::init() { };

void AExpression::init(std::vector<ATerm> _terms, bool _simplify=true, bool _sort=true) {
    terms = _terms;

    if (_simplify) {
        simplify();
    }
    if (_sort) {
        sort();
    }
}

void AExpression::init(Expression ex, bool _simplify=true, bool _sort=true) {
    for (unsigned int i = 0; i < ex.terms.size(); i++) {
        ATerm term;
        term.init(ex.terms[i]);
        terms.push_back(term);
    }

    if (_simplify) {
        simplify();
    }
    if (_sort) {
        sort();
    }
}

void AExpression::simplify() {
    std::vector<ATerm> newterms;
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (abs(terms[i].scalar) > tthresh) {
            newterms.push_back(terms[i]);
        }
    }

    std::vector<bool> keep;
    for (unsigned int i = 0; i < newterms.size(); i++) {
        keep[i] = true;
    }

    for (unsigned int i = 0; i < newterms.size(); i++) {
        if (!(keep[i])) {
            continue;
        }

        for (unsigned int j = 0; j < newterms.size(); j++) {
            if ((!(keep[j])) || (i == j)) {
                continue;
            }

            int sign = newterms[i].pmatch(newterms[j]);

            if (sign != 0) {
                newterms[i].scalar += sign * newterms[j].scalar;
                keep[j] = false;
            }
        }
    }

    terms.clear();

    for (unsigned int i = 0; i < newterms.size(); i++) {
        if (keep[i] and (abs(newterms[i].scalar) > tthresh)) {
            terms.push_back(newterms[i]);
        }
    }
}

void AExpression::sort_tensors() {
    for (unsigned int i = 0; i < terms.size(); i++) {
        terms[i].sort_tensors();
    }
}

// FIXME
void AExpression::sort() {
    unsigned int x, y, min;
    for (x = 0; x < terms.size()-1; x++) {
        for (y = x; y < terms.size(); y++) {
            if (terms[y] < terms[min]) {
                min = y;
            }
            swap(terms[x], terms[min]);
        }
    }
}

bool AExpression::connected() {
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (!(terms[i].connected())) {
            return false;
        }
    }

    return true;
}

AExpression AExpression::get_connected(bool _simplify=true, bool _sort=true) {
    std::vector<ATerm> newterms;
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (terms[i].connected()) {
            newterms.push_back(terms[i]);
        }
    }

    AExpression out;
    out.init(newterms, _simplify, _sort);

    return out;
}

bool AExpression::pmatch(AExpression other) {
    if (terms.size() != other.terms.size()) {
        return false;
    }

    for (unsigned int i = 0; i < terms.size(); i++) {
        bool matched = false;

        for (unsigned int j = 0; j < other.terms.size(); j++) {
            if (other.terms[j].pmatch(terms[i])) {
                matched = true;
                break;
            }
        }

        if (!(matched)) {
            return false;
        }
    }

    return true;
}

void AExpression::transpose(std::vector<int> perm) {
    for (unsigned int i = 0; i < terms.size(); i++) {
        terms[i].transpose(perm);
    }
}





int main() {
    Idx a;
    a.init(0, "o", true);

    Projector p;
    p.init(a);

    FOperator fop;
    fop.init(a, true);

    Idx b;
    b.init(0, "o", false);

    BOperator bop;
    bop.init(b, true);

    std::vector<std::vector<int>> plist;
    plist.push_back({0, 1, 2, 3});
    plist.push_back({0, 1});

    std::vector<int> signs;
    signs.push_back(1);
    signs.push_back(-1);

    TensorSym ts;
    ts.init(plist, signs);

    std::vector<Sigma> sums;
    std::vector<Tensor> tensors;

    for (int i = 0; i < 4; i++) {
        Sigma s;
        if (i % 2 == 0) {
            s.init(a);
        } else {
            s.init(b);
        }
        sums.push_back(s);

        Tensor t;
        t.init({a, b}, std::to_string(i), ts);
        tensors.push_back(t);
    }

    TermMap tm;
    tm.init(sums, tensors);

    return 0;
}
