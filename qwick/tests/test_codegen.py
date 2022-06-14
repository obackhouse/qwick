import unittest
import itertools

from fractions import Fraction

from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import one_e, two_e, E1, E2, braE1, braE2, commute
from qwick import codegen

import sympy

import numpy as np

from pyscf import gto, scf, cc, ao2mo, lib

sizes = {
        "nocc": sympy.Symbol("N"),
        "nvir": sympy.Symbol("N") * 5,
}


class CodegenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.bra1, cls.bra2, cls.Hbar = bra1, bra2, Hbar

        cls.particles = {
                "f": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
                "v": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
                "t1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
                "t2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
                "t1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
                "t2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        }

        cls.printer = codegen.EinsumPrinter(
                occupancy_tags={
                    "v": "{base}.{tags}",
                    "f": "{base}.{tags}",
                },
                reorder_axes={
                    "v": (0, 2, 1, 3),
                    "t1": (1, 0),
                    "t2": (2, 3, 0, 1),
                    "t1new": (1, 0),
                    "t2new": (2, 3, 0, 1),
                },
                remove_spacing=True,
                garbage_collection=True,
                base_indent=0,
                einsum="np.einsum",
                zeros="np.zeros",
                dtype="np.float64",
        )

        cls.mol = gto.Mole()
        cls.mol.atom = "O 0 0 0; O 0 0 1"
        cls.mol.basis = "6-31g"
        cls.mol.atom = "He 0 0 0"
        cls.mol.basis = "cc-pvdz"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.ccsd = cc.CCSD(cls.mf)
        cls.ccsd.max_cycle = 1
        cls.ccsd.kernel()

        cls.nocc = np.sum(cls.mf.mo_occ > 0)
        cls.nvir = np.sum(cls.mf.mo_occ == 0)

        fock = np.linalg.multi_dot((cls.mf.mo_coeff.conj().T, cls.mf.get_fock(), cls.mf.mo_coeff))
        cls.f = lambda: None
        cls.f.oo = fock[:cls.nocc, :cls.nocc] - np.diag(cls.mf.mo_energy[:cls.nocc])
        cls.f.ov = fock[:cls.nocc, cls.nocc:]
        cls.f.vo = fock[cls.nocc:, :cls.nocc]
        cls.f.vv = fock[cls.nocc:, cls.nocc:] - np.diag(cls.mf.mo_energy[cls.nocc:])

        cls.v = lambda: None
        for tup in itertools.product(("o", "v"), repeat=4):
            coeffs = [cls.mf.mo_coeff[:, cls.mf.mo_occ > 0 if k == "o" else cls.mf.mo_occ == 0] for k in tup]
            eri = ao2mo.incore.general(cls.mf._eri, coeffs, compact=False)
            eri = eri.reshape([c.shape[-1] for c in coeffs])
            setattr(cls.v, "".join(tup), eri)

    @classmethod
    def tearDownClass(cls):
        del cls.bra1, cls.bra2, cls.Hbar, cls.particles, cls.f, cls.v, cls.printer, cls.mol, cls.mf, cls.ccsd, cls.nocc, cls.nvir

    def test_ccsd_energy(self):
        expr = apply_wick(self.Hbar)
        expr.resolve()
        expr = AExpression(Ex=expr)

        terms, indices = codegen.wick_to_sympy(expr, self.particles, return_value="e")
        terms = codegen.ghf_to_rhf(terms, indices)
        eqns = [codegen.sympy_to_drudge(terms, indices)]
        eqns_opt = codegen.optimize(
                eqns,
                sizes=sizes,
                optimize="exhaust",
                verify=True,
                interm_fmt="x{}",
        )

        ccsd = self.ccsd
        f, v, nocc, nvir = self.f, self.v, self.nocc, self.nvir
        t1, t2 = ccsd.t1, ccsd.t2

        ref = self.ccsd.energy(t1=t1, t2=t2, eris=ccsd.ao2mo())

        exec(self.printer.doprint(eqns))
        self.assertAlmostEqual(np.abs(ref - locals()["e"]), 0.0, 8)

        exec(self.printer.doprint(eqns_opt))
        self.assertAlmostEqual(np.abs(ref - locals()["e"]), 0.0, 8)

    def test_ccsd_t1(self):
        expr = apply_wick(self.bra1 * self.Hbar)
        expr.resolve()
        expr = AExpression(Ex=expr)

        terms, indices = codegen.wick_to_sympy(expr, self.particles, return_value="t1new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        eqns = [codegen.sympy_to_drudge(terms, indices)]
        eqns_opt = codegen.optimize(
                eqns,
                sizes=sizes,
                optimize="exhaust",
                verify=True,
                interm_fmt="x{}",
        )

        ccsd = self.ccsd
        f, v, nocc, nvir = self.f, self.v, self.nocc, self.nvir
        t1, t2 = ccsd.t1, ccsd.t2
        e_ia = lib.direct_sum("i-a->ia", self.mf.mo_energy[:nocc], self.mf.mo_energy[nocc:])

        ref = self.ccsd.update_amps(t1=t1, t2=t2, eris=ccsd.ao2mo())[0]

        exec(self.printer.doprint(eqns))
        locals()["t1new"] /= e_ia
        self.assertAlmostEqual(np.max(np.abs(ref - locals()["t1new"])), 0.0, 8)

        exec(self.printer.doprint(eqns_opt))
        locals()["t1new"] /= e_ia
        self.assertAlmostEqual(np.max(np.abs(ref - locals()["t1new"])), 0.0, 8)

    def test_ccsd_t2(self):
        expr = apply_wick(self.bra2 * self.Hbar)
        expr.resolve()
        expr = AExpression(Ex=expr)

        terms, indices = codegen.wick_to_sympy(expr, self.particles, return_value="t2new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        eqns = [codegen.sympy_to_drudge(terms, indices)]
        eqns_opt = codegen.optimize(
                eqns,
                sizes=sizes,
                optimize="greedy",
                verify=False,
                interm_fmt="x{}",
        )

        ccsd = self.ccsd
        f, v, nocc, nvir = self.f, self.v, self.nocc, self.nvir
        t1, t2 = ccsd.t1, ccsd.t2
        e_ia = lib.direct_sum("i-a->ia", self.mf.mo_energy[:nocc], self.mf.mo_energy[nocc:])
        e_ijab = lib.direct_sum("ia,jb->ijab", e_ia, e_ia)

        ref = self.ccsd.update_amps(t1=t1, t2=t2, eris=ccsd.ao2mo())[1]

        with open("tmp.dat", "w") as out:
            out.write(self.printer.doprint(eqns))
        exec(self.printer.doprint(eqns))
        locals()["t2new"] /= e_ijab
        self.assertAlmostEqual(np.max(np.abs(ref - locals()["t2new"])), 0.0, 8)

        exec(self.printer.doprint(eqns_opt))
        locals()["t2new"] /= e_ijab
        self.assertAlmostEqual(np.max(np.abs(ref - locals()["t2new"])), 0.0, 8)



if __name__ == "__main__":
    if codegen is not None:
        unittest.main()
