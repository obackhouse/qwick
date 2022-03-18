from fractions import Fraction
from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import one_e, E1, braE1, commute

H1 = one_e("f", ["occ", "vir"], norder=True)

bra = braE1("occ", "vir")
T1 = E1("t", ["occ"], ["vir"])

HT = commute(H1, T1)
HTT = commute(HT, T1)
S = bra*(H1 + HT + Fraction('1/2')*HTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
