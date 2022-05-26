"""Convert between `wick` and `drudge` representations.
"""

from typing import Union
import drudge
import sympy

AExpression = []
try:
    from wick.expression import AExpression as AExpr
    AExpression.append(AExpr)
except:
    pass
try:
    from qwick.expression import AExpression as AExpr
    AExpression.append(AExpr)
except:
    pass
AExpression = Union[tuple(AExpression)]


def wick_to_drudge(expr: AExpression, dr: drudge.Drudge = None, return_value: str = "res"):
    """Convert from a `wick.AExpression` to a `drudge.Tensor`.

    Arguments
    ---------
    expr: wick.AExpression
        Expression from `wick`.
    dr: drudge.Drudge, optional
        `drudge` object, will be initialised using a dummy `pyspark`
        context if `None`. Default value is `None`.
    return_value: str, optional
        Return tensor, default value is "res".

    Returns
    -------
    expr: drudge.TensorDef
        Tensor definition in `drudge` format.
    groups: PermType
        Groups giving permutations of tensors.
    """

    terms = expr.terms
    group = {}
    drudge_expr = 0

    # Keys are [fermion, space]
    index_lists = {
            #(True, "occ"): ["o%d" % i for i in range(100)],
            #(True, "vir"): ["v%d" % i for i in range(100)],
            #(False, "occ"): ["O%d" % i for i in range(100)],
            #(False, "vir"): ["V%d" % i for i in range(100)],
            (True, "occ"): ["i", "j", "k", "l", "m", "n"] + ["o%d" % n for n in range(25)],
            (True, "vir"): ["a", "b", "c", "d", "e", "f"] + ["v%d" % n for n in range(25)],
            (False, "occ"): ["I", "J", "K", "L", "M", "N"] + ["O%d" % n for n in range(25)],
            (False, "vir"): ["A", "B", "C", "D", "E", "F"] + ["V%d" % n for n in range(25)],
    }

    # Keys are [fermion, space, summed, index]
    indices = {}

    # The following contain no more information than the indices dict,
    # they are just included for convenience:
    # Keys are [fermion, space]
    sectors = {(fermion, space): set() for fermion in (True, False) for space in ("occ", "vir")}
    has_fermion = {fermion: False for fermion in (True, False)}

    # Get all the base symbols ahead of time to make sure there is no
    # conflict with index names:
    symbol_bases = set()
    for term in terms:
        for tensor in term.tensors:
            base = sympy.IndexedBase(tensor.name)
            symbol_bases.add(base)

    def idx_to_string(idx, summed):
        """Converts a `wick.Idx` to a string consistently across
        numerous terms.
        """

        if (idx.fermion, idx.space, summed, idx.index) not in indices:
            next_idx = index_lists[idx.fermion, idx.space].pop(0)
            while next_idx in symbol_bases:
                next_idx = index_lists[idx.fermion, idx.space].pop(0)
            indices[idx.fermion, idx.space, summed, idx.index] = next_idx
            sectors[idx.fermion, idx.space].add(next_idx)
            has_fermion[idx.fermion] = True

        return indices[idx.fermion, idx.space, summed, idx.index]

    # Get the input tensors and indices:
    sympy_exprs = []
    return_exprs = []
    groups = {}
    for term in terms:
        # Get the scalar - qwick doesn't currently support fractions
        # so check if this is exactly 1 within tthresh  # FIXME - 1e-15 is too low
        if abs(term.scalar - 1) < 1e-14:
            sympy_expr = 1
        elif abs(term.scalar + 1) < 1e-14:
            sympy_expr = -1
        else:
            sympy_expr = term.scalar

        return_expr = None
        summed = set(s.idx for s in term.sums)
        for tensor in term.tensors:
            if tensor.name != "":
                # Get tensor and indices:
                base = sympy.IndexedBase(tensor.name)
                inds = []
                for index in tensor.indices:
                    inds.append(sympy.Symbol(idx_to_string(index, index in summed)))
                sympy_expr *= base[tuple(inds)]

                # Get group entry:
                if base not in groups:
                    perms = []
                    for perm, sign in tensor.sym.tlist:
                        acc = drudge.IDENT if sign == 1 else drudge.NEG
                        perms.append((perm, acc))
                    groups[(base, len(inds))] = tuple(perms)

            else:
                # This is the output tensor
                base = sympy.IndexedBase(return_value)
                inds = []
                for index in tensor.indices:
                    inds.append(sympy.Symbol(idx_to_string(index, False)))
                return_expr = base[tuple(inds)]

        sympy_exprs.append(sympy_expr)
        return_exprs.append(return_expr)

    # All return exprs should be the same:
    assert all(return_exprs[0] == r for r in return_exprs)
    return_expr = return_exprs[0]

    # If no return tensor was found, it has no indices:
    return_expr = sympy.Symbol(return_value)

    # Initialise drudge:
    if dr is None:
        from dummy_spark import SparkContext
        ctx = SparkContext()
        dr = drudge.Drudge(ctx)

    # Declare spaces:
    if has_fermion[True]:
        nfocc = sympy.Symbol("nfocc")
        nfvir = sympy.Symbol("nfvir")
        focc = drudge.Range("focc", 0, nfocc)
        fvir = drudge.Range("fvir", 0, nfvir)
    if has_fermion[False]:
        nbocc = sympy.Symbol("nbocc")
        nbvir = sympy.Symbol("nbvir")
        bocc = drudge.Range("bocc", 0, nbocc)
        bvir = drudge.Range("bvir", 0, nbvir)

    # Declare dummy indices:
    if has_fermion[True]:
        dr.set_dumms(focc, tuple(sectors[True, "occ"]))
        dr.set_dumms(fvir, tuple(sectors[True, "vir"]))
    if has_fermion[False]:
        dr.set_dumms(bocc, tuple(sectors[False, "occ"]))
        dr.set_dumms(bvir, tuple(sectors[False, "vir"]))
    dr.add_resolver_for_dumms()

    # Build einstein summations:
    expr = dr.einst(sympy_exprs[0])
    for sympy_expr in sympy_exprs[1:]:
        expr += dr.einst(sympy_expr)

    # Build TensorDef:
    tensordef = dr.define(return_expr, expr)

    return tensordef, groups


qwick_to_drudge = wick_to_drudge



if __name__ == "__main__":
    # Get the wick expression:
    from fractions import Fraction
    from wick.expression import AExpression
    from wick.wick import apply_wick
    from wick.convenience import one_e, two_e, E1, E2, braE1, commute

    H1 = one_e("f", ["occ", "vir"], norder=True)
    H2 = two_e("I", ["occ", "vir"], norder=True)
    H = H1 + H2

    bra = braE1("occ", "vir")
    T1 = E1("t", ["occ"], ["vir"])
    T2 = E2("t", ["occ"], ["vir"])
    T = T1 + T2

    HT = commute(H, T)
    HTT = commute(HT, T)
    HTTT = commute(commute(commute(H2, T1), T1), T1)

    S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)

    # Get the drudge expression:
    from dummy_spark import SparkContext  # TODO

    ctx = SparkContext()
    dr = drudge.Drudge(ctx)

    expr, groups = wick_to_drudge(final, dr=dr)
    print(expr)
