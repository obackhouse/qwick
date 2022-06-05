"""Convert between formats.
"""

import sympy
try:
    import drudge
except:
    drudge = None
from sympy import S
from qwick.codegen.spin_integrate import *


def wick_to_sympy(expr, particles: dict, return_value: str = "res"):
    """Convert from a `wick.AExpression` to a `Term`.
    """

    convert_space = lambda space: {"occ": OCCUPIED, "vir": VIRTUAL, "nm": BOSON}[space]
    index_lists_full = {
            OCCUPIED: ["i", "j", "k", "l", "m", "n", "o", "p"] + ["o%d" % n for n in range(25)],
            VIRTUAL:  ["a", "b", "c", "d", "e", "f", "g", "h"] + ["v%d" % n for n in range(25)],
            BOSON:    ["w", "x", "y", "z"] + ["b%d" % n for n in range(25)],
    }
    index_lists = {key: val.copy() for key, val in index_lists_full.items()}

    # Keys are [space, summed, index]
    indices = {}

    # Convenience:
    sectors = {space: set() for space in index_lists.keys()}
    has_fermions = False
    has_bosons = False

    # Get all the tensor names ahead of time to make sure there is no
    # conflict with index names:
    symbol_bases = set()
    for term in expr.terms:
        for tensor in term.tensors:
            symbol_bases.add(tensor.name)

    # Function to consistently convert a `wick.Idx` to a string:
    def idx_to_string(idx, summed):
        if (idx.space, summed, idx.index) not in indices:
            next_idx = index_lists[convert_space(idx.space)].pop(0)
            indices[idx.space, summed, idx.index] = next_idx
            sectors[convert_space(idx.space)].add(next_idx)
            if idx.fermion:
                has_fermions = True
            else:
                has_bosons = True
        return indices[idx.space, summed, idx.index]

    # Get the input tensors and indices:
    terms = []
    for term in sorted(expr.terms):
        rhs = []
        lhs = None

        # Get the scalar - qwick doesn't currently support fractions
        # so check within a threshold
        # FIXME
        if abs(term.scalar - 1) < 1e-14:
            rhs.append(S.One)
        elif abs(term.scalar + 1) < 1e-14:
            rhs.append(S.NegativeOne)
        else:
            rhs.append(term.scalar)

        return_expr = None
        dummies = set(s.idx for s in term.sums)
        externals = set()
        for tensor in term.tensors:
            for index in tensor.indices:
                if index not in dummies:
                    externals.add(index)

        for tensor in term.tensors:
            if tensor.name != "":
                # Get TensorSymbol name and rank:
                base_name = tensor.name
                rank = len(tensor.indices)

                # Get indices:
                inds = []
                for index in tensor.indices:
                    name = idx_to_string(index, index in dummies)
                    space = convert_space(index.space)
                    cls = DummyIndex if index in dummies else ExternalIndex
                    inds.append(cls(name, space))

                # Get group entry:
                group = []
                for perm, sign in tensor.sym.tlist:
                    acc = IDENTITY if sign == 1 else NEGATIVE
                    group.append((perm, acc))

                # Build TensorSymbol:
                symb = TensorSymbol(base_name, rank, group=group, particles=particles[base_name])

                # Build Tensor:
                tensor = symb[tuple(inds)]
                rhs.append(tensor)

        for tensor in term.tensors:
            if tensor.name == "":
                # Get output TensorSymbol name and rank:
                base_name = return_value
                rank = len(externals)

                if rank != 0:
                    # Get output indices:
                    inds = []
                    for index in tensor.indices:
                        name = idx_to_string(index, False)
                        space = convert_space(index.space)
                        inds.append(ExternalIndex(name, space))

                    # Build output TensorSymbol:
                    symb = TensorSymbol(base_name, rank)

                    # Build Tensor:
                    tensor = symb[tuple(inds)]
                    lhs = tensor

                else:
                    # Build output Symbol:
                    symb = sympy.Symbol(base_name)
                    lhs = symb

        if lhs is None:
            lhs = sympy.Symbol(return_value)
            assert len(externals) == 0

        terms.append(Term(lhs, rhs))

    # Get dummies:
    dummies = {
        OCCUPIED: [
            DummyIndex(name, convert_space(space))
            for (space, summed, index), name in indices.items()
            if (summed and convert_space(space) == OCCUPIED)
        ],
        VIRTUAL: [
            DummyIndex(name, convert_space(space))
            for (space, summed, index), name in indices.items()
            if (summed and convert_space(space) == VIRTUAL)
        ],
        BOSON: [
            DummyIndex(name, convert_space(space))
            for (space, summed, index), name in indices.items()
            if (summed and convert_space(space) == BOSON)
        ],
    }

    # Get externals:
    externals = {
        OCCUPIED: [
            ExternalIndex(name, convert_space(space))
            for (space, summed, index), name in indices.items()
            if (not summed and convert_space(space) == OCCUPIED)
        ],
        VIRTUAL: [
            ExternalIndex(name, convert_space(space))
            for (space, summed, index), name in indices.items()
            if (not summed and convert_space(space) == VIRTUAL)
        ],
        BOSON: [
            ExternalIndex(name, convert_space(space))
            for (space, summed, index), name in indices.items()
            if (not summed and convert_space(space) == BOSON)
        ],
    }

    # Add more dummies:
    for space in (OCCUPIED, VIRTUAL, BOSON):
        for name in index_lists[space]:
            index = DummyIndex(name, space)
            dummies[space].append(index)

    # Build full index dictionary:
    indices = build_indices_dict(dummies, externals)

    return terms, indices


def sympy_to_drudge(terms, indices, dr=None):
    """Convert from a list of `Term` to `drudge.TensorDef`.
    """
    # TODO doesn't support UHF

    # Initialise drudge:
    if dr is None:
        from dummy_spark import SparkContext
        ctx = SparkContext()
        dr = drudge.Drudge(ctx)

    # Declare spaces:
    nocc = sympy.Symbol("nocc")
    nvir = sympy.Symbol("nvir")
    nbos = sympy.Symbol("nbos")
    occ = drudge.Range("occ", 0, nocc)
    vir = drudge.Range("vir", 0, nvir)
    bos = drudge.Range("bos", 0, nbos)

    # Declare dummy indices:
    for rng, space in [(occ, OCCUPIED), (vir, VIRTUAL), (bos, BOSON)]:
        dumms = [sympy.Symbol(index.name) for index in indices["dummies"][space, None]]
        dr.set_dumms(rng, dumms)
    dr.add_resolver_for_dumms()

    # Build einstein summations:
    expr = 0
    groups = {}
    for term in terms:
        part = term.factor
        for tensor in term.rhs_tensors:
            base = sympy.IndexedBase(tensor.base.name)
            indices = [sympy.Symbol(index.name) for index in tensor.indices]
            part *= base[tuple(indices)]
            if base not in groups:
                groups[base] = tensor.group
        expr += part

    # Build TensorDef:
    assert all(terms[0].lhs == term.lhs for term in terms)
    if isinstance(terms[0].lhs, Tensor):
        lhs = [sympy.IndexedBase(terms[0].lhs.base.name)]
        for index in terms[0].lhs.indices:
            lhs.append((
                sympy.Symbol(index.name),
                {OCCUPIED: occ, VIRTUAL: vir, BOSON: bos}[index.space],
            ))
    else:
        lhs = [sympy.Symbol(terms[0].lhs.name)]
    tensordef = dr.define(*lhs, dr.einst(expr))

    # Set symmetry:
    for symb, group in groups.items():
        perms = []
        for perm, action in group:
            if tuple(perm) != tuple(range(len(perm))):
                acc = {
                    IDENTITY: drudge.IDENT,
                    NEGATIVE: drudge.NEG,
                    CONJUGATE: drudge.CONJ,
                }[action]
                perms.append(drudge.Perm(perm, acc))
        if len(perms) == 0:
            perms = [None]
        dr.set_symm(symb, *perms)

    # Simplify the TensorDef:
    tensordef = tensordef.simplify()

    return tensordef



if __name__ == "__main__":
    from fractions import Fraction
    from qwick.expression import AExpression
    from qwick.wick import apply_wick
    from qwick.convenience import one_e, two_e, E1, E2, braE1, commute

    H1 = one_e("f", ["occ", "vir"], norder=True)
    H2 = two_e("v", ["occ", "vir"], norder=True)
    H = H1 + H2

    bra = braE1("occ", "vir")
    T1 = E1("t1", ["occ"], ["vir"])
    T2 = E2("t2", ["occ"], ["vir"])
    T = T1 + T2

    HT = commute(H, T)
    HTT = commute(HT, T)
    HTTT = commute(commute(commute(H2, T1), T1), T1)
    H = (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)

    out = apply_wick(bra * H)
    out.resolve()
    final = AExpression(Ex=out)

    print(final)
    print()

    particles = {
            "f": [(FERMION, 0), (FERMION, 0)],
            "v": [(FERMION, 0), (FERMION, 1), (FERMION, 0), (FERMION, 1)],
            "t1": [(FERMION, 0), (FERMION, 0)],
            "t2": [(FERMION, 0), (FERMION, 1), (FERMION, 0), (FERMION, 1)],
    }

    terms, indices = wick_to_sympy(final, particles)

    terms = spin_orbital_to_restricted(terms, indices)

    expr = sympy_to_drudge(terms, indices)
