"""Optimize tensor contractions for a series of expressions.
"""

from typing import List, Tuple, Dict
import drudge
import gristmill
import sympy


def optimize(
        eqns: List[drudge.TensorDef],
        sizes: Dict[drudge.Range, int],
        optimize: str = "exhaust",
        verify: bool = True,
        interm_fmt: str = "x{}",
        **kwargs,
):
    """Optimize the tensor contractions for a list of equations. Each
    equation must have a LHS and a definition therein. Symmetries in
    each symbol should be set ahead of calling this function.

    Arguments
    ---------
    eqns: list of drudge.TensorDef
        Definitions of each expression.
    sizes: dictionary of (drudge.Range, int)
        Sizes for each `drudge.Range`, which are substituted in order
        to facilitate optimisation of the contractions.
    optimize: str, optional
        Optimisation strategy, can be one of {"greedy", "opt", "trav",
        "exhaust"}. For specific details of each strategy refer to the
        `gristmill` documentation; generally they should improve in
        quality of the final expressions but degrade in efficiency of
        finding the final expressions in the order listed. Default
        value is "exhaust".
    verify: bool, optional
        If True, verify that the optimised evaluation sequence is
        symbollically equivalent to the input. May be slow for large
        examples but is recommended in the `gristmill` documentation.
    interm_fmt: str, optional
        Format of intermediate values. Default value is `x{}`.
    """

    strat = getattr(gristmill.ContrStrat, optimize.upper())
    eqns_opt = gristmill.optimize(
            eqns,
            substs=sizes,
            contr_strat=strat,
            interm_fmt=interm_fmt,
            **kwargs,
    )

    if verify:
        assert gristmill.verify_eval_seq(eqns_opt, eqns, simplify=True)

    return eqns_opt



if __name__ == "__main__":
    from dummy_spark import SparkContext

    ctx = SparkContext()
    dr = drudge.Drudge(ctx)

    # Declare variables:
    v = sympy.IndexedBase("v")

    # Declare space sizes:
    nocc = sympy.Symbol("nocc")
    nvir = sympy.Symbol("nvir")

    # Declare space ranges:
    occ = drudge.Range("occ", 0, nocc)
    vir = drudge.Range("vir", 0, nvir)

    # Declare indices:
    i, j, k, l = sympy.symbols("i:l")
    a, b, c, d = sympy.symbols("a:d")

    # Set dummy indices:
    dr.set_dumms(occ, (i, j))
    dr.set_dumms(vir, (a, b))
    dr.add_resolver_for_dumms()

    # Declare groups:
    groups = {
            v: (
                ([0, 1, 2, 3], drudge.IDENT),
                ([1, 0, 3, 2], drudge.IDENT),
            ),
    }

    # Define expression:
    expr  = dr.einst(v[i, j, a, b] * v[a, b, i, j]) * 2.0
    expr -= dr.einst(v[i, j, a, b] * v[a, b, j, i])
    eqns = [dr.define(sympy.Symbol("e"), expr)]

    # Optimise:
    eqns_opt, (flops_orig, flops_opt) = optimize(eqns, {nocc: 100, nvir: 500}, groups)

    for eqn in eqns:
        print(eqn)
    print()
    for eqn in eqns_opt:
        print(eqn)
