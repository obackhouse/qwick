from fractions import Fraction

import convert
import spin_integrate
import optimize
import printer

from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import one_e, two_e, E1, E2, braE2, commute

import sympy

import drudge
import gristmill
import dummy_spark


# --- Generate a statement using (q)wick:

h1 = one_e("f", ["occ", "vir"], norder=True)
h2 = two_e("v", ["occ", "vir"], norder=True, compress=True)
h = h1 + h2

bra = braE2("occ", "vir", "occ", "vir")
t1 = E1("t1", ["occ"], ["vir"])
t2 = E2("t2", ["occ"], ["vir"])
t = t1 + t2

ht = commute(h, t)
htt = commute(ht, t)
httt = commute(htt, t)
#htttt = commute(httt, t)
hbar = h + ht + Fraction(1, 2) * htt
hbar += Fraction(1, 6) * httt
#hbar += Fraction(1, 24) * htttt

out = apply_wick(hbar)
out.resolve()

expr = AExpression(Ex=out)


# --- Convert to a format compatible with drudge/gristmill:

ctx = dummy_spark.SparkContext()
dr = drudge.Drudge(ctx)

expr, groups = convert.wick_to_drudge(expr, dr=dr)
eqns = [expr]


# --- Perform spin-integration on the expression:

particles = {
        (sympy.IndexedBase("f"), 2): [(drudge.FERMI, 0), (drudge.FERMI, 0)],
        (sympy.IndexedBase("v"), 4): [(drudge.FERMI, 0), (drudge.FERMI, 1), (drudge.FERMI, 0), (drudge.FERMI, 1)],
        (sympy.IndexedBase("t1"), 2): [(drudge.FERMI, 0), (drudge.FERMI, 0)],
        (sympy.IndexedBase("t2"), 4): [(drudge.FERMI, 0), (drudge.FERMI, 1), (drudge.FERMI, 0), (drudge.FERMI, 1)],
}

rhf_eqns = []
for eqn in eqns:
    rhf_eqns += spin_integrate.spin_integrate(dr, eqn, groups, particles, restricted=True)


# --- Perform the optimisation:

sizes = {
        sympy.Symbol("nfocc"): 100,
        sympy.Symbol("nfvir"): 500,
}

opt_rhf_eqns = optimize.optimize(
        rhf_eqns,
        sizes=sizes,
        optimize="exhaust",
        verify=True,
        interm_fmt="x{}",
)


# --- Print einsums:

printer = printer.EinsumPrinter(
        zeros="np.zeros",
        einsum="lib.einsum",
        dtype="np.float64",
        base_indent=0,
        occupancy_tags={
            "f": "{base}.{tags}",
            "v": "{base}.{tags}",
        },
        reorder_axes={
            "v": (0, 2, 1, 3),  # Chemist notation
            "t1": (1, 0),       # Occ,vir instead of vir,occ
            "t2": (2, 3, 0, 1), # As above
        },
        remove_spacing=True,
)

print()
print(printer.doprint(opt_rhf_eqns))
