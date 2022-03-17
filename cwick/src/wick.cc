/*
 *  Wick theorem
 */

#include "operator.h"
#include "expression.h"
#include "wick.h"


// TODO occ keyword
bool valid_contraction(Operator o1, Operator o2) {
    if (o1.idx.space != o2.idx.space) {
        return false;
    }

    if (o1.fermion && o2.fermion) {
        if (is_occupied(o1.idx) && o1.ca && (!(o2.ca))) {
            return true;
        }
        if (!(is_occupied(o1.idx)) && (!(o1.ca)) && o2.ca) {
            return true;
        }
        return false;
    }
    else if (!(o1.fermion) && !(o2.fermion)) {
        if (!(o1.ca) && o2.ca) {
            return true;
        }
        return false;
    }
    else if (o1.fermion == o2.fermion) {
        return true;
    }

    return false;
}

// TODO occ keyword
std::vector<std::vector<std::pair<Operator, Operator>>> pair_list(std::vector<Operator> lst) {
    int n = lst.size();
    assert(n % 2 == 0);

    if (n < 2) {
        return {};
    }
    else if (n == 2) {
        if (valid_contraction(lst[0], lst[1])) {
            std::pair<Operator, Operator> p;
            p.first = lst[0];
            p.second = lst[1];
            return {{p}};
        }
        else {
            return {};
        }
    }
    else {
        std::vector<std::vector<std::pair<Operator, Operator>>> plist;
        std::vector<Operator> ltmp(lst.begin() + 1, lst.end());
        for (unsigned int i = 0; i < ltmp.size(); i++) {
            if (valid_contraction(lst[0], ltmp[i])) {
                std::pair<Operator, Operator> p1;
                p1.first = lst[0];
                p1.second = ltmp[i];

                std::vector<Operator> lrem;
                for (unsigned int j = 0; j < i; j++) {
                    lrem.push_back(ltmp[j]);
                }
                for (unsigned int j = i+1; j < ltmp.size(); j++) {
                    lrem.push_back(ltmp[j]);
                }

                std::vector<std::vector<std::pair<Operator, Operator>>> rem = pair_list(lrem);

                for (unsigned int j = 0; j < rem.size(); j++) {
                    rem[j].push_back(p1);
                    plist.push_back(rem[j]);
                }
            }
        }
        return plist;
    }
}

// pair(-1,-1) returned instead of None
std::pair<int, int> find_pair(int i, std::vector<std::pair<int, int>> ipairs) {
    for (unsigned int p = 0; p < ipairs.size(); p++) {
        if ((ipairs[p].first == i) || (ipairs[p].second == i)) {
            return ipairs[p];
        }
    }

    std::pair<int, int> p;
    p.first = -1;
    p.second = -1;

    return p;
}
        
int get_sign(std::vector<std::pair<int, int>> ipairs) {
    int ncross = 0;

    for (unsigned int p = 0; p < ipairs.size(); p++) {
        int i = ipairs[p].first;
        int j = ipairs[p].second;

        for (int x1 = i+1; x1 < j; x1++) {
            std::pair<int, int> p1 = find_pair(x1, ipairs);

            if ((p1.first == -1) && (p1.second == -1)) {
                continue;
            }

            int x2;
            if (p1.second == x1) {
                x2 = p1.first;
            } else {
                x2 = p1.second;
            }

            if ((x2 > j) || (x2 < i)) {
                ncross += 1;
            }
        }
    }

    assert(ncross % 2 == 0);
    ncross = ncross / 2;

    int out;
    if (ncross % 2 == 0) {
        out = 1;
    } else {
        out = -1;
    }
    return out;
}

std::vector<std::vector<Operator>> split_operators(std::vector<Operator> ops) {
    std::vector<int> ps;
    for (unsigned int i = 0; i < ops.size(); i++) {
        if (ops[i].projector) {
            ps.push_back(i);
        }
    }

    if (ps.size() == 0) {
        return {ops};
    }

    std::vector<int> starts = {0};
    for (unsigned int i = 0; i < ps.size(); i++) {
        starts.push_back(ps[i] + 1);
    }

    std::vector<int> ends(ps);
    ends.push_back(ops.size());

    std::vector<std::vector<Operator>> olists;
    assert(starts.size() == ends.size());
    for (unsigned int i = 0; i < starts.size(); i++) {
        int s = starts[i];
        int e = ends[i];
        std::vector<Operator> olists_i;
        for (int j = s; j < e; j++) {
            olists_i.push_back(ops[j]);
        }
        olists.push_back(olists_i);
    }

    return olists;
}

// TODO occ keyword
Expression apply_wick(Expression e) {
    std::vector<Term> to;

    for (unsigned int i = 0; i < e.terms.size(); i++) {
        Term temp = e.terms[i];
        std::vector<std::vector<Operator>> olists = split_operators(temp.operators);

        bool any_olists = false;
        for (unsigned int j = 0; j < olists.size(); j++) {
            if (olists[j].size() != 0) {
                any_olists = true;
                break;
            }
        }
        if (!(any_olists)) {
            to.push_back(temp.copy());
            continue;
        }

        bool all_oparity = true;
        for (unsigned int j = 0; j < olists.size(); j++) {
            if (olists[j].size() % 2 != 0) {
                all_oparity = false;
                break;
            }
        }
        if (!(all_oparity)) {
            continue;
        }

        std::vector<std::vector<std::vector<Delta>>> dos;
        std::vector<std::vector<int>> sos;

        for (unsigned int j = 0; j < olists.size(); j++) {
            if (olists[j].size() == 0) {
                continue;
            }

            auto plist = pair_list(olists[j]);
            std::vector<std::vector<Delta>> ds;
            std::vector<int> ss;

            for (unsigned int k = 0; k < plist.size(); k++) {
                bool good = plist[k].size() != 0;

                std::vector<std::pair<int, int>> ipairs;
                std::vector<Delta> deltas;

                for (unsigned int l = 0; l < plist[k].size(); l++) {
                    auto p = plist[k][l];
                    auto oi = p.first;
                    auto oj = p.second;

                    if (oi.idx.space != oj.idx.space) {
                        good = false;
                        break;
                    }
                    if (!(oi.idx.fermion)) {
                        Idx i1 = oi.idx;
                        Idx i2 = oj.idx;
                        deltas.push_back(Delta(i1, i2));
                    }
                    else if ((is_occupied(oi.idx) && oi.ca && (!(oj.ca))) ||
                             ((!(is_occupied(oi.idx))) && (!(oi.ca)) && oj.ca)) {
                        unsigned int ii = olists[j].size();
                        for (ii = 0; ii < olists[j].size(); ii++) {
                            if (oi == olists[j][ii]) {
                                break;
                            }
                        }
                        assert(ii != olists[j].size());
                        unsigned int jj = olists[j].size();
                        for (jj = 0; jj < olists[j].size(); jj++) {
                            if (oj == olists[j][jj]) {
                                break;
                            }
                        }
                        assert(ii != olists[j].size());

                        std::pair<int, int> p;
                        p.first = ii;
                        p.second = jj;
                        ipairs.push_back(p);

                        Idx i1 = oi.idx;
                        Idx i2 = oj.idx;
                        deltas.push_back(Delta(i1, i2));
                    }
                    else {
                        good = false;
                        break;
                    }
                }

                if (good) {
                    ds.push_back(deltas);
                    ss.push_back(get_sign(ipairs));
                }
            }

            dos.push_back(ds);
            sos.push_back(ss);
        }

        assert(sos.size() == dos.size());
        // Loop over a cartesian product of elements of dos and sos
        // https://stackoverflow.com/questions/5279051
        auto product = [](long long a, std::vector<int>& b) {
            return a*b.size();
        };
        const long long N = std::accumulate(sos.begin(), sos.end(), 1LL, product);
        std::vector<std::vector<Delta>> dos_prod(sos.size());
        std::vector<int> sos_prod(sos.size());

        for (long long j = 0; j < N; j++) {
            std::lldiv_t q {j, 0};
            for (long long k = sos.size()-1; k >= 0; k--) {
                q = std::div(q.quot, sos[k].size());
                sos_prod[k] = sos[k][q.rem];
                dos_prod[k] = dos[k][q.rem];
            }

            // assert sos_prod
            // assert dos_prod

            int sign = 1;
            for (unsigned int k = 0; k < sos_prod.size(); k++) {
                sign *= sos_prod[k];
            }

            std::vector<Delta> deltas;
            for (unsigned int k = 0; k < dos_prod.size(); k++) {
                for (unsigned int l = 0; l < dos_prod[k].size(); l++) {
                    deltas.push_back(dos_prod[k][l]);
                }
            }

            double scalar = sign * temp.scalar;

            std::vector<Sigma> sums;
            std::vector<Tensor> tensors;
            std::vector<Operator> operators;

            for (auto k = temp.sums.begin(); k < temp.sums.end(); k++) {
                sums.push_back((*k).copy());
            }
            for (auto k = temp.tensors.begin(); k < temp.tensors.end(); k++) {
                tensors.push_back((*k).copy());
            }
            for (auto k = temp.deltas.begin(); k < temp.deltas.end(); k++) {
                deltas.push_back((*k).copy());
            }

            Term t1(scalar, sums, tensors, operators, deltas, default_index_key());;
            to.push_back(t1);
        }
    }

    Expression o(to);
    assert(!(o.are_operators()));

    return o;
}
