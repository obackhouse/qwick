from fractions import Fraction

from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import one_e, two_e, E1, E2, braE1, braE2, commute
from qwick import codegen

import sympy

import drudge
import gristmill
import pyspark
import dummy_spark


# Build the wick statement:
H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("v", ["occ", "vir"], norder=True)
H = H1 + H2
bra1 = braE1("occ", "vir")
bra2 = braE2("occ", "vir", "occ", "vir")
T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])  # Operators with different ranks must have different names
T = T1 + T2
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT
out = apply_wick(Hbar)
out.resolve()
expr = AExpression(Ex=out)

# Convert to sympy format:
particles = {
        "f": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "v": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "t1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
}
terms, indices = codegen.wick_to_sympy(expr, particles)

# Perform spin integration:
terms = codegen.ghf_to_rhf(terms, indices)
for term in terms:
    print(term)

# Convert to drudge format:
expr = codegen.sympy_to_drudge(terms, indices)

# Optimise contractions:
sizes = {"nocc": sympy.Symbol("N"), "nvir": sympy.Symbol("N")*5}
eqns = codegen.optimize(
        [expr],
        sizes=sizes,
        optimize="exhaust",
        verify=False,
        interm_fmt="x{}",
)

# Generate code:
printer = codegen.EinsumPrinter(
        occupancy_tags={
            "v": "{base}.{tags}",
            "f": "{base}{tags}",
        },
        reorder_axes={
            "v": (0, 2, 1, 3),   # Chemist's notation
            "t1": (1, 0),        # Occupied first
            "t2": (2, 3, 0, 1),  # Occupied both first
        },
        remove_spacing=True,     # Remove blank lines
        garbage_collection=True, # Delete intermediates after use
        base_indent=0,           # No indentation
        einsum="lib.einsum",     # Use PySCF einsum function
        # n.b. the following are defunct with explicit_init=False:
        zeros="np.zeros",        # NumPy initialisation
        dtype="np.float64",      # Data type
)
print(printer.doprint(eqns))

# Print number of FLOPs:
print(gristmill.get_flop_cost(eqns))
