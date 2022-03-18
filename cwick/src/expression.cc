/*
 *  Expressions
 */

#include "util.h"
#include "index.h"
#include "operator.h"
#include "expression.h"

#include <math.h>


TermMap::TermMap() {}

TermMap::TermMap(const std::vector<Sigma> &sums, const std::vector<Tensor> &tensors) {
    std::vector<Idx> sindices(sums.size());
    for (unsigned int i = 0; i < sums.size(); i++) {
        sindices[i] = sums[i].idx;
    }

    for (auto ti = tensors.begin(); ti < tensors.end(); ti++) {
        std::vector<Idx> ilist = (*ti).ilist();
        std::string tiname = ((*ti).name == "") ? "!" : (*ti).name;

        // can't be arsed with a dict, use two vectors instead FIXME
        // TODO place these keys in a sorted fashion
        std::vector<std::string> strs_keys;
        std::vector<std::unordered_set<std::string>> strs_vals;
        for (auto x = ilist.begin(); x < ilist.end(); x++) {
            bool found = false;
            for (auto y = strs_keys.begin(); y < strs_keys.end(); y++) {
                if ((*y) == (*x).space) {
                    found = true;
                    break;
                }
            }
            if (!(found)) {
                strs_keys.push_back((*x).space);
                strs_vals.push_back(std::unordered_set<std::string>());
            }
        }

        for (unsigned int i = 0; i < (*ti).indices.size(); i++) {
            const Idx *iidx = &((*ti).indices[i]);
            std::string space = (*iidx).space;
            std::string istr = std::to_string(i);

            bool summed = false;
            for (auto s = sindices.begin(); s < sindices.end(); s++) {
                if ((*s) == (*iidx)) {
                    summed = true;
                    break;
                }
            }

            for (auto tj = tensors.begin(); tj < tensors.end(); tj++) {
                std::string tjname = ((*tj).name == "") ? "!" : (*tj).name;

                // jts
                std::unordered_set<std::pair<std::string, Idx>, IdxStringPairHash> jts;
                for (unsigned int j = 0; j < (*tj).indices.size(); j++) {
                    const Idx *jidx = &((*tj).indices[j]);

                    if ((*iidx) == (*jidx)) {
                        std::pair<std::string, Idx> p(std::to_string(j), *jidx);
                        jts.insert(p);
                    }
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
                    assert((strs_idx != -1) && (strs_idx != strs_keys.size()));

                    //std::unordered_set<std::pair<std::string, Idx>, IdxStringPairHash>::iterator itr;
                    for (auto itr = jts.begin(); itr != jts.end(); ++itr) {
                        // make_str
                        std::string x_str = istr + tjname + (*itr).first;
                        if (summed) {
                            x_str += "x";
                        }
                        strs_vals[strs_idx].insert(x_str);
                    }
                }
            }
        }

        // lll: sort strs by the keys
        // strs: strs_keys (vector of str) : strs_vals (vector of set<str>)
        auto mask = argsort(strs_keys);
        strs_keys = apply_argsort(strs_keys, mask);
        strs_vals = apply_argsort(strs_vals, mask);

        // make element of TermMap.data
        std::unordered_set<TERM_MAP_SUB_DATA_TYPE, TermMapSubDataHash> subdatas;
        for (unsigned int x = 0; x < strs_keys.size(); x++) {
            TERM_MAP_SUB_DATA_TYPE p(strs_keys[x], strs_vals[x]);
            subdatas.insert(p);
        }

        TERM_MAP_DATA_TYPE elem(tiname, subdatas);
        data.insert(elem);
    }
}

bool operator==(const TERM_MAP_SUB_DATA_TYPE &a, const TERM_MAP_SUB_DATA_TYPE &b) {
    if (a.first != b.first) {
        return false;
    } else if (a.second.size() != b.second.size()) {
        return false;
    } else {
        for (auto ai = a.second.begin(); ai != a.second.end(); ai++) {
            if (b.second.find(*ai) == b.second.end()) {
                return false;
            }
        }
    }

    return true;
}

bool operator!=(const TERM_MAP_SUB_DATA_TYPE &a, const TERM_MAP_SUB_DATA_TYPE &b) {
    return (!(a == b));
}

bool operator==(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b) {
    if (a.first != b.first) {
        return false;
    } else if (a.second.size() != b.second.size()) {
        return false;
    } else {
        for (auto ai = a.second.begin(); ai != a.second.end(); ai++) {
            if (b.second.find(*ai) == b.second.end()) {
                return false;
            }
        }
        return true;
    }
}

bool operator!=(const TERM_MAP_DATA_TYPE &a, const TERM_MAP_DATA_TYPE &b) {
    return (!(a == b));
}

bool operator==(const TermMap &a, const TermMap &b) {
    if (a.data.size() != b.data.size()) {
        return false;
    } else {
        for (auto ai = a.data.begin(); ai != a.data.end(); ai++) {
            if (b.data.find(*ai) == b.data.end()) {
                return false;
            }
        }
        return true;
    }
}

bool operator!=(const TermMap &a, const TermMap &b) {
    return (!(a == b));
}


int get_case(const Delta &dd, const std::unordered_set<Idx, IdxHash> &ilist) {
    assert(dd.i1.space == dd.i2.space);

    std::unordered_set<Idx, IdxHash>::const_iterator it1 = ilist.find(dd.i1);
    bool is1 = (it1 != ilist.end());

    std::unordered_set<Idx, IdxHash>::const_iterator it2 = ilist.find(dd.i2);
    bool is2 = (it2 != ilist.end());

    int _case = 0;
    if (is1) {
        _case = 1;
    }
    if (is2) {
        _case = (_case == 0) ? 2 : 3;
    }

    return _case;
}

void _resolve(std::vector<Sigma> &sums,
              std::vector<Tensor> &tensors,
              std::vector<Operator> &operators,
              std::vector<Delta> &deltas) {
    std::vector<Sigma> newsums(sums.size());
    std::vector<Tensor> newtensors(tensors.size());
    std::vector<Operator> newoperators(operators.size());
    std::vector<Delta> newdeltas;

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums[i] = sums[i].copy();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors[i] = tensors[i].copy();
    }
    for (unsigned int i = 0; i < operators.size(); i++) {
        newoperators[i] = operators[i].copy();
    }

    for (unsigned int i = 0; i < deltas.size(); i++) {
        bool found = false;
        for (unsigned int j = 0; j < newdeltas.size(); j++) {
            if (deltas[i] == newdeltas[j]) {
                found = true;
                break;
            }
        }
        if (!(found)) {
            newdeltas.push_back(deltas[i].copy());
        }
    }

    std::unordered_set<Idx, IdxHash> ilist;
    for (unsigned int i = 0; i < sums.size(); i++) {
        ilist.insert(sums[i].idx);
    }

    std::vector<int> cases(newdeltas.size());
    for (unsigned int d = 0; d < newdeltas.size(); d++) {
        cases[d] = get_case(newdeltas[d], ilist);
    }

    std::vector<Delta> rs;

    // Iterate over a copy of cases
    std::vector<int> cases1(cases);
    for (unsigned int d = 0; d < newdeltas.size(); d++) {
        int _case = cases1[d];
        const Idx i1 = newdeltas[d].i1;
        const Idx i2 = newdeltas[d].i2;

        if (_case == 3) {
            continue;
        }
        if (_case == 1) {
            Sigma sigma_i1(i1);
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
            Sigma sigma_i2(i2);
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
                    throw std::runtime_error("Invalid case in _resolve [1]");
                }
            } else if ((_case == 1) && (newdeltas[i].i2 == i1)) {
                newdeltas[i].i2 = i2;
                if (ccc == 3) {
                    cases[i] = 1;
                } else if (ccc == 2) {
                    cases[i] = 0;
                } else {
                    throw std::runtime_error("Invalid case in _resolve [2]");
                }
            } else if ((_case == 2) && (newdeltas[i].i2 == i2)) {
                newdeltas[i].i2 = i1;
                if (ccc == 3) {
                    cases[i] = 1;
                } else if (ccc == 2) {
                    cases[i] = 0;
                } else {
                    throw std::runtime_error("Invalid case in _resolve [3]");
                }
            } else if ((_case == 2) && (newdeltas[i].i1 == i2)) {
                newdeltas[i].i1 = i1;
                if (ccc == 3) {
                    cases[i] = 2;
                } else if (ccc == 1) {
                    cases[i] = 0;
                } else {
                    throw std::runtime_error("Invalid case in _resolve [4]");
                }
            }
        }

        for (auto t = newtensors.begin(); t < newtensors.end(); t++) {
            for (auto k = (*t).indices.begin(); k < (*t).indices.end(); k++) {
                if (_case == 1) {
                    if ((*k) == i1) {
                        (*k) = i2;
                    }
                } else if (_case == 2) {
                    if ((*k) == i2) {
                        (*k) = i1;
                    }
                }
            }
        }

        for (auto o = newoperators.begin(); o < newoperators.end(); o++) {
            if (_case == 1) {
                if ((*o).idx == i1) {
                    (*o).idx = i2;
                }
            } else if (_case == 2) {
                if ((*o).idx == i2) {
                    (*o).idx = i1;
                }
            }
        }

        if (!((_case == 0) && (i1 != i2))) {
            rs.push_back(newdeltas[d]);
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
            sums = newsums;
            tensors = newtensors;
            operators = newoperators;
            deltas = newdeltas;
            return;
        }
    }

    rs.clear();

    assert(newdeltas.size() == cases.size());
    for (unsigned int d = 0; d < newdeltas.size(); d++) {
        int _case = cases[d];
        const Idx i1 = newdeltas[d].i1;
        const Idx i2 = newdeltas[d].i2;

        if (_case == 3) {
            Sigma sigma_i2(i2);
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
            throw std::runtime_error("Invalid case in _resolve [5]");
        }

        if (_case == 0) {
            continue;
        }

        for (auto t = newtensors.begin(); t < newtensors.end(); t++) {
            for (auto k = (*t).indices.begin(); k < (*t).indices.end(); k++) {
                if ((*k) == i2) {
                    (*k) = i1;
                }
            }
        }

        for (auto o = newoperators.begin(); o < newoperators.end(); o++) {
            if ((*o).idx == i2) {
                (*o).idx = i1;
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

const std::unordered_map<std::string, std::string> default_index_key = {
    {"occ", "ijklmnop"},
    {"vir", "abcdefgh"},
    {"nm", "IJKLMNOP"},
};

Term::Term() {
    scalar = 0.0;
}

Term::Term(
        const double _scalar,
        const std::vector<Sigma> &_sums,
        const std::vector<Tensor> &_tensors,
        const std::vector<Operator> &_operators,
        const std::vector<Delta> &_deltas,
        const std::unordered_map<std::string, std::string> _index_key) {
    scalar = _scalar;
    sums = _sums;
    tensors = _tensors;
    operators = _operators;
    deltas = _deltas;
    index_key = _index_key;
}

void Term::resolve() {
    _resolve(sums, tensors, operators, deltas);
}

std::string Term::repr() const {
    std::string out = format_float(scalar, false);

    for (unsigned int i = 0; i < sums.size(); i++) {
        out += sums[i].repr();
    }
    for (unsigned int i = 0; i < deltas.size(); i++) {
        out += deltas[i].repr();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        out += tensors[i].repr();
    }
    for (unsigned int i = 0; i < operators.size(); i++) {
        out += operators[i].repr();
    }

    return out;
}

std::string Term::_print_str(const bool with_scalar) const {
    auto imap = _idx_map();

    std::string out = "";
    if (with_scalar) {
        out += format_float(scalar, false);
    }

    for (auto x = sums.begin(); x < sums.end(); x++) {
        out += (*x)._print_str(imap);
    }
    for (auto x = deltas.begin(); x < deltas.end(); x++) {
        out += (*x)._print_str(imap);
    }
    for (auto x = tensors.begin(); x < tensors.end(); x++) {
        out += (*x)._print_str(imap);
    }
    for (auto x = operators.begin(); x < operators.end(); x++) {
        out += (*x)._print_str(imap);
    }

    return out;
}

std::unordered_map<Idx, std::string, IdxHash> Term::_idx_map() const {
    auto _ilist = ilist();

    std::unordered_map<std::string, int> off;
    std::unordered_map<Idx, std::string, IdxHash> imap;

    int o;
    for (auto idx = _ilist.begin(); idx < _ilist.end(); idx++) {
        if (off.find((*idx).space) != off.end()) {
            o = off[(*idx).space];
            off[(*idx).space] += 1;
        } else {
            o = 0;
            off[(*idx).space] = 1;
        }

        imap[*idx] = index_key.at((*idx).space)[o];
    }

    return imap;
}

std::vector<Idx> Term::ilist() const {
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

    std::sort(out.begin(), out.end());

    return out;
}

Term Term::_inc(const int i) const {
    std::vector<Sigma> newsums(sums.size());
    std::vector<Tensor> newtensors(tensors.size());
    std::vector<Operator> newoperators(operators.size());
    std::vector<Delta> newdeltas(deltas.size());

    for (unsigned int j = 0; j < sums.size(); j++) {
        newsums[j] = sums[j]._inc(i);
    }
    for (unsigned int j = 0; j < tensors.size(); j++) {
        newtensors[j] = tensors[j]._inc(i);
    }
    for (unsigned int j = 0; j < operators.size(); j++) {
        newoperators[j] = operators[j]._inc(i);
    }
    for (unsigned int j = 0; j < deltas.size(); j++) {
        newdeltas[j] = deltas[j]._inc(i);
    }

    Term out(scalar, newsums, newtensors, newoperators, newdeltas, index_key);

    return out;
}

Term Term::copy() const {
    double newscalar = scalar;
    std::vector<Sigma> newsums(sums.size());
    std::vector<Tensor> newtensors(tensors.size());
    std::vector<Operator> newoperators(operators.size());
    std::vector<Delta> newdeltas(deltas.size());

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums[i] = sums[i].copy();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors[i] = tensors[i].copy();
    }
    for (unsigned int i = 0; i < operators.size(); i++) {
        newoperators[i] = operators[i].copy();
    }
    for (unsigned int i = 0; i < deltas.size(); i++) {
        newdeltas[i] = deltas[i].copy();
    }

    Term out(newscalar, newsums, newtensors, newoperators, newdeltas, index_key);

    return out;
}

Term operator*(const Term &a, const Term &b) {
    const std::vector<Idx> il1 = a.ilist();
    const std::vector<Idx> il2 = b.ilist();
    std::unordered_set<Idx, IdxHash> sil1, sil2, sil12_intersection;

    for (unsigned int i = 0; i < il1.size(); i++) {
        sil1.insert(il1[i]);
    }
    for (unsigned int i = 0; i < il2.size(); i++) {
        sil2.insert(il2[i]);
    }
    for (std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
         itr != sil1.end();
         itr++) {
        if (sil2.find(*itr) != sil2.end()) {
            sil12_intersection.insert(*itr);
        }
    }

    Term newb;
    if (sil12_intersection.size() != 0) {
        int m = -1;
        for (std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
             itr != sil1.end();
             itr++) {
            m = ((*itr).index > m) ? (*itr).index : m;
        }
        assert(m != -1);
        newb = b._inc(m + 1);
    } else {
        newb = b;
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

    if (a.index_key == default_index_key) {
        new_index_key = b.index_key;
    } else {
        new_index_key = a.index_key;
    }

    Term newterm(newscalar, newsums, newtensors, newoperators, newdeltas, new_index_key);

    return newterm;
}

Term operator*(const Term &a, const double &b) {
    Term newterm = a.copy();
    newterm.scalar *= b;
    return newterm;
}

Term operator*(const double &a, const Term &b) {
    return (b * a);
}

Term operator*(const Term &a, const int &b) {
    Term newterm = a.copy();
    newterm.scalar = newterm.scalar * b;
    return newterm;
}

Term operator*(const int &a, const Term &b) {
    return (b * a);
}

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

ATerm::ATerm() {}

ATerm::ATerm(
            const double _scalar,
            const std::vector<Sigma> &_sums,
            const std::vector<Tensor> &_tensors,
            const std::unordered_map<std::string, std::string> _index_key) {
    scalar = _scalar;
    sums = _sums;
    tensors = _tensors;
    index_key = _index_key;
}

ATerm::ATerm(const Term &term) {
    scalar = term.scalar;
    index_key = term.index_key;

    assert(term.operators.size() == 0);

    sums.reserve(term.sums.size());
    tensors.reserve(term.tensors.size() + term.deltas.size());

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

std::string ATerm::repr() const {
    std::string out = format_float(scalar, false);

    for (unsigned int i = 0; i < sums.size(); i++) {
        out += sums[i].repr();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        out += tensors[i].repr();
    }

    return out;
}

std::string ATerm::_print_str(const bool with_scalar) const {
    auto imap = _idx_map();

    std::string out = "";
    if (with_scalar) {
        out += format_float(scalar, true);
    }

    std::string iis = "";
    for (auto x = sums.begin(); x < sums.end(); x++) {
        iis += imap.at((*x).idx);
    }
    if (iis != "") {
        out += "\\sum_{" + iis + "}";
    }

    for (auto x = tensors.begin(); x < tensors.end(); x++) {
        out += (*x)._print_str(imap);
    }

    return out;
}

std::string ATerm::_einsum_str() const {
    auto imap = _idx_map();
    std::string sstr = format_float(scalar, true);
    std::string fstr = "";
    std::string istr = "";
    std::string tstr = "";

    for (auto tt = tensors.begin(); tt < tensors.end(); tt++) {
        if ((*tt).name == "") {
            fstr += (*tt)._istr(imap);
        } else {
            tstr += ", " + (*tt).name;
            istr += (*tt)._istr(imap) + ",";
        }
    }

    istr.pop_back();

    return sstr + "*einsum('" + istr + "->" + fstr + "'" + tstr + ")";
}

std::unordered_map<Idx, std::string, IdxHash> ATerm::_idx_map() const {
    auto _ilist = ilist();

    std::unordered_map<std::string, int> off;
    std::unordered_map<Idx, std::string, IdxHash> imap;

    int o;
    for (auto idx = _ilist.begin(); idx < _ilist.end(); idx++) {
        if (off.find((*idx).space) != off.end()) {
            o = off.at((*idx).space);
            off[(*idx).space] += 1;
        } else {
            o = 0;
            off[(*idx).space] = 1;
        }

        imap[*idx] = index_key.at((*idx).space)[o];
    }

    return imap;
}

ATerm ATerm::_inc(const int i) const {
    std::vector<Sigma> newsums(sums.size());
    std::vector<Tensor> newtensors(tensors.size());

    for (unsigned int j = 0; j < sums.size(); j++) {
        newsums[j] = sums[j]._inc(i);
    }
    for (unsigned int j = 0; j < tensors.size(); j++) {
        newtensors[j] = tensors[j]._inc(i);
    }

    ATerm out(scalar, newsums, newtensors, index_key);

    return out;
}

bool ATerm::match(const ATerm other) const {
    TermMap TM1(sums, tensors);
    TermMap TM2(other.sums, other.tensors);
    return TM1 == TM2;
}

// Returns 0 instead of None
int ATerm::pmatch(const ATerm other) const {
    if (sums.size() != other.sums.size()) {
        return 0;
    }
    if (tensors.size() != other.tensors.size()) {
        return 0;
    }

    std::vector<std::vector<std::pair<std::vector<int>, int>>> tlists(other.tensors.size());
    for (unsigned int i = 0; i < other.tensors.size(); i++) {
        tlists[i] = other.tensors[i].sym.tlist;
    }

    TermMap TM1(sums, tensors);

    // Loop over a cartesian product of elements of tlists
    // https://stackoverflow.com/questions/5279051
    auto product = [](long long a, std::vector<std::pair<std::vector<int>, int>>& b) {
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
        for (auto x = u.begin(); x < u.end(); x++) {
            sign *= (*x).second;
        }

        // permute
        assert(other.tensors.size() == u.size());
        std::vector<Tensor> newtensors(other.tensors.size());
        for (unsigned int x = 0; x < other.tensors.size(); x++) {
            newtensors[x] = permute(other.tensors[x], u[x].first);
        }

        TermMap TM2(other.sums, newtensors);

        if (TM1 == TM2) {
            return sign;
        }
    }

    return 0;
}

// FIXME scales badly
std::vector<Idx> ATerm::ilist() const {
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

unsigned int ATerm::nidx() const {
    const std::vector<Idx> idxs = ilist();
    return idxs.size();
}

void ATerm::sort_tensors() {
    unsigned int off = 0;
    for (unsigned int i = 0; i < tensors.size(); i++) {
        if (tensors[i].name == "") {
            auto temp = tensors[i];
            tensors[i] = tensors[off];
            tensors[off] = temp;
            off++;
        }
    }
}

void ATerm::merge_external() {
    bool ext = true;

    for (unsigned int t = 0; t > tensors.size(); t++) {
        if ((!(ext)) && (tensors[t].name == "")) {
            assert(0);
        }
        if (tensors[t].name != "") {
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
        std::vector<Tensor> newtensors(tensors.size() - num_ext);
        for (unsigned int t = num_ext; t < tensors.size(); t++) {
            newtensors[t-num_ext] = tensors[t].copy();
        }

        std::vector<Idx> ext_indices;
        for (unsigned int t = 0; t < num_ext; t++) {
            for (unsigned int i = 0; i < tensors[t].indices.size(); i++) {
                ext_indices.push_back(tensors[t].indices[i]);
            }
        }

        newtensors.insert(newtensors.begin(), Tensor(ext_indices, ""));

        tensors = newtensors;
    }
}

bool ATerm::connected() const {
    std::vector<Idx> ll(sums.size());
    for (unsigned int i = 0; i < sums.size(); i++) {
        ll[i] = sums[i].idx;
    }

    std::vector<Tensor> rtensors;
    for (unsigned int i = 0; i < tensors.size(); i++) {
        if ((tensors[i].name != "") && (tensors[i].indices.size() != 0)) {
            rtensors.push_back(tensors[i]);
        }
    }

    std::vector<std::unordered_set<int>> adj(ll.size());
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

        for (auto b = blue.begin(); b != blue.end(); b++) {
            for (auto ad = adj.begin(); ad != adj.end(); ad++) {
                if (std::find((*ad).begin(), (*ad).end(), (*b)) != (*ad).end()) {
                    for (auto a = (*ad).begin(); a != (*ad).end(); a++) {
                        newtensors.push_back(*a);
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

bool ATerm::reducible() const {
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
        for (auto t = newterm.tensors.begin(); t < newterm.tensors.end(); t++) {
            for (auto x = (*t).indices.begin(); x < (*t).indices.end(); x++) {
                if ((*x) == i1) {
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

void ATerm::transpose(const std::vector<int> &perm) {
    merge_external();
    tensors[0].transpose(perm);
}

ATerm ATerm::copy() const {
    std::vector<Tensor> newtensors(tensors.size());
    std::vector<Sigma> newsums(sums.size());

    for (unsigned int i = 0; i < sums.size(); i++) {
        newsums[i] = sums[i].copy();
    }
    for (unsigned int i = 0; i < tensors.size(); i++) {
        newtensors[i] = tensors[i].copy();
    }

    ATerm out(scalar, newsums, newtensors, index_key);

    return out;
}

ATerm operator*(const ATerm &a, const ATerm &b) {
    const std::vector<Idx> il1 = a.ilist();
    const std::vector<Idx> il2 = b.ilist();
    std::unordered_set<Idx, IdxHash> sil1, sil2, sil12_intersection;

    for (unsigned int i = 0; i < il1.size(); i++) {
        sil1.insert(il1[i]);
    }
    for (unsigned int i = 0; i < il2.size(); i++) {
        sil2.insert(il2[i]);
    }
    for (std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
         itr != sil1.end();
         itr++) {
        if (sil2.find(*itr) != sil2.end()) {
            sil12_intersection.insert(*itr);
        }
    }

    ATerm newb;
    if (sil12_intersection.size() != 0) {
        int m = -1;
        for (std::unordered_set<Idx, IdxHash>::iterator itr = sil1.begin();
             itr != sil1.end();
             itr++) {
            m = ((*itr).index > m) ? (*itr).index : m;
        }
        assert(m != -1);
        newb = b._inc(m + 1);
    } else {
        newb = b;
    }

    double newscalar = a.scalar * newb.scalar;

    std::vector<Sigma> newsums = a.sums;
    std::vector<Tensor> newtensors = a.tensors;

    newsums.insert(newsums.end(), newb.sums.begin(), newb.sums.end());
    newtensors.insert(newtensors.end(), newb.tensors.begin(), newb.tensors.end());

    std::unordered_map<std::string, std::string> new_index_key;

    if (a.index_key == default_index_key) {
        new_index_key = b.index_key;
    } else {
        new_index_key = a.index_key;
    }

    ATerm newterm(newscalar, newsums, newtensors, new_index_key);

    return newterm;
}

ATerm operator*(const ATerm &a, const double &b) {
    ATerm newterm = a.copy();
    newterm.scalar *= b;
    return newterm;
}

ATerm operator*(const double &a, const ATerm &b) {
    return (b * a);
}

ATerm operator*(const ATerm &a, const int &b) {
    ATerm newterm = a.copy();
    newterm.scalar = newterm.scalar * b;
    return newterm;
}

ATerm operator*(const int &a, const ATerm &b) {
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

Expression::Expression() {}

Expression::Expression(const std::vector<Term> &_terms) {
    terms = _terms;
}

void Expression::resolve() {
    for (unsigned int i = 0; i < terms.size(); i++) {
        terms[i].resolve();
    }

    // TODO for loops like this, allocate the full vector and then resize?
    std::vector<Term> newterms;
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (fabs(terms[i].scalar) > tthresh) {
            newterms.push_back(terms[i]);
        }
    }

    terms = newterms;
}

std::string Expression::repr() const {
    std::string out = "";

    for (unsigned int i = 0; i < terms.size(); i++) {
        out += terms[i].repr();
        if (i != (terms.size()-1)) {
            out += " + ";
        }
    }

    return out;
}

std::string Expression::_print_str() const {
    std::string out = "";

    for (auto t = terms.begin(); t < terms.end(); t++) {
        if (out != "") {
            out += "\n";
        }
        std::string sign = ((*t).scalar < 0) ? " - " : " + ";
        out += sign + format_float(fabs((*t).scalar), false) + (*t)._print_str(false);
    }

    return out;
}

bool Expression::are_operators() const {
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (terms[i].operators.size() > 0) {
            return true;
        }
    }

    return false;
}

Expression operator+(const Expression &a, const Expression &b) {
    std::vector<Term> newterms(a.terms.size() + b.terms.size());
    for (unsigned int i = 0; i < a.terms.size(); i++) {
        newterms[i] = a.terms[i];
    }
    for (unsigned int i = 0; i < b.terms.size(); i++) {
        newterms[i+a.terms.size()] = b.terms[i];
    }

    return Expression(newterms);
}

Expression operator-(const Expression &a, const Expression &b) {
    double fac = -1.0;
    Expression mb = b * fac;
    return (a + mb);
}

Expression operator*(const Expression &a, const Expression &b) {
    std::vector<Term> newterms(a.terms.size() * b.terms.size());

    for (unsigned int i = 0, ij = 0; i < a.terms.size(); i++) {
        for (unsigned int j = 0; j < b.terms.size(); j++, ij++) {
            newterms[ij] = a.terms[i] * b.terms[j];
        }
    }

    return Expression(newterms);
}

Expression operator*(const Expression &a, const double &b) {
    std::vector<Term> newterms(a.terms.size());

    for (unsigned int i = 0; i < a.terms.size(); i++) {
        newterms[i] = a.terms[i] * b;
    }

    return Expression(newterms);
}

Expression operator*(const double &a, const Expression &b) {
    return (b * a);
}

Expression operator*(const Expression &a, const int &b) {
    std::vector<Term> newterms(a.terms.size());

    for (unsigned int i = 0; i < a.terms.size(); i++) {
        newterms[i] = a.terms[i] * b;
    }

    return Expression(newterms);
}

Expression operator*(const int &a, const Expression &b) {
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


AExpression::AExpression() {}

AExpression::AExpression(const std::vector<ATerm> &_terms, const bool _simplify, const bool _sort) {
    terms = _terms;

    if (_simplify) {
        simplify();
    }
    if (_sort) {
        sort();
    }
}

AExpression::AExpression(const Expression &ex, const bool _simplify, const bool _sort) {
    terms.reserve(ex.terms.size());
    for (unsigned int i = 0; i < ex.terms.size(); i++) {
        ATerm term(ex.terms[i]);
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
    // TODO prealloc
    std::vector<ATerm> newterms;
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (fabs(terms[i].scalar) > tthresh) {
            newterms.push_back(terms[i]);
        }
    }

    std::vector<bool> keep(newterms.size(), true);

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
        int x = 0;
        for (unsigned int j = 0; j < keep.size(); j++) {
            if (keep[j]) {
                x += 1;
            }
        }
    }

    terms.clear();

    for (unsigned int i = 0; i < newterms.size(); i++) {
        if (keep[i] && (fabs(newterms[i].scalar) > tthresh)) {
            terms.push_back(newterms[i]);
        }
    }
}

void AExpression::sort_tensors() {
    for (unsigned int i = 0; i < terms.size(); i++) {
        terms[i].sort_tensors();
    }
}

void AExpression::sort() {
    std::sort(terms.begin(), terms.end());
}

std::string AExpression::_print_str() const {
    std::string out = "";

    for (auto t = terms.begin(); t < terms.end(); t++) {
        if (out != "") {
            out += "\n";
        }
        std::string sign = ((*t).scalar < 0) ? " - " : " + ";
        out += sign + format_float(fabs((*t).scalar), true) + (*t)._print_str(false);
    }

    return out;
}

std::string AExpression::_print_einsum(const std::string lhs) const {
    std::string out = "";

    for (auto t = terms.begin(); t < terms.end(); t++) {
        if (out != "") {
            out += "\n";
        }
        out += lhs + " += " + (*t)._einsum_str();
    }

    return out;
}

bool AExpression::connected() const {
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (!(terms[i].connected())) {
            return false;
        }
    }

    return true;
}

AExpression AExpression::get_connected(const bool _simplify, const bool _sort) const {
    // TODO prealloc
    std::vector<ATerm> newterms;
    for (unsigned int i = 0; i < terms.size(); i++) {
        if (terms[i].connected()) {
            newterms.push_back(terms[i]);
        }
    }

    return AExpression(newterms, _simplify, _sort);
}

bool AExpression::pmatch(const AExpression &other) const {
    if (terms.size() != other.terms.size()) {
        return false;
    }

    for (auto t1 = terms.begin(); t1 < terms.end(); t1++) {
        bool matched = false;
        for (auto t2 = other.terms.begin(); t2 < other.terms.end(); t2++) {
            if ((*t2).pmatch(*t1)) {
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

void AExpression::transpose(const std::vector<int> &perm) {
    for (unsigned int i = 0; i < terms.size(); i++) {
        terms[i].transpose(perm);
    }
}
