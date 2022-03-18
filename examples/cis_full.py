from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import one_e, two_e, E1, braE1

H1 = one_e("f", ["occ", "vir"])
H2 = two_e("I", ["occ", "vir"])

H = H1 + H2
bra = braE1("occ", "vir")
ket = E1("c", ["occ"], ["vir"])

HC = H*ket
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
