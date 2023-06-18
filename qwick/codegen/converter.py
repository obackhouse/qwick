"""Convert between formats.
"""

import drudge
import sympy
import itertools
from sympy import S
from qwick.codegen.spin_integrate import *


def wick_to_sympy(expr, particles: dict, skip_symmetry=set(), return_value: str = "res"):
    """Convert from a `wick.AExpression` to a list of `Term`s.
    """

    spaces = (OCCUPIED, VIRTUAL, BOSON)
    convert_space = lambda space: {"occ": OCCUPIED, "vir": VIRTUAL, "nm": BOSON}[space]
    index_lists = {
            OCCUPIED: ["i", "j", "k", "l", "m", "n", "o", "p", "I", "J", "K", "L", "M", "N", "O", "P"],
            VIRTUAL:  ["a", "b", "c", "d", "e", "f", "g", "h", "A", "B", "C", "D", "E", "F", "G", "H"],
            BOSON:    ["s", "t", "u", "v", "w", "x", "y", "z", "S", "T", "U", "V", "W", "X", "Y", "Z"],
    }

    # Remove any names which conflict with index names:
    for space in spaces:
        if return_value in index_lists[space]:
            index_lists[space].remove(return_value)
        for term in expr.terms:
            for tensor in term.tensors:
                if tensor.name in index_lists[space]:
                    index_lists[space].remove(tensor.name)

    # Assign the externals
    externals = {space: [] for space in spaces}
    wick_externals = set()
    for term in expr.terms:
        summed_indices = {s.idx for s in term.sums}
        for tensor in term.tensors:
            for index in tensor.indices:
                if index not in summed_indices and index not in wick_externals:
                    wick_externals.add(index)
                    space = convert_space(index.space)
                    name = index_lists[space].pop(0)
                    index = ExternalIndex(name, space)
                    externals[space].append(index)

    # Assign the dummies
    dummies = {space: [] for space in spaces}
    for space in spaces:
        while len(index_lists[space]):
            name = index_lists[space].pop(0)
            index = DummyIndex(name, space)
            dummies[space].append(index)

    # Get the input tensors and indices:
    terms = []
    for term in sorted(expr.terms):
        rhs = []
        lhs = None

        # Get the scalar
        scalar = term.scalar
        if abs(scalar - 1) < 1e-14:
            scalar = S.One
        elif abs(scalar + 1) < 1e-14:
            scalar = S.NegativeOne
        rhs.append(scalar)

        # Maps between wick and sympy indices
        externals_copy = {space: externals[space].copy() for space in spaces}
        dummies_copy = {space: dummies[space].copy() for space in spaces}
        externals_map = {}
        dummies_map = {}
        summed_indices = {s.idx for s in term.sums}

        for tensor in term.tensors:
            if tensor.name != "":
                # Get TensorSymbol name and rank
                base_name = tensor.name
                rank = len(tensor.indices)

                # Get indices:
                inds = []
                for index in tensor.indices:
                    space = convert_space(index.space)
                    if index in summed_indices:
                        if index not in dummies_map:
                            dummies_map[index] = dummies_copy[space].pop(0)
                        inds.append(dummies_map[index])
                    else: 
                        if index not in externals_map:
                            externals_map[index] = externals_copy[space].pop(0)
                        inds.append(externals_map[index])

                # Get group entry:
                group = []
                if base_name not in skip_symmetry:
                    for perm, sign in tensor.sym.tlist:
                        acc = IDENTITY if sign == 1 else NEGATIVE
                        group.append((perm, acc))

                # Build TensorSymbol:
                symb = TensorSymbol(base_name, rank, group, particles[base_name])

                # Build Tensor:
                tensor = symb[tuple(inds)]
                rhs.append(tensor)

        # If we didn't find an input tensor, must determine the external indices:
        if len(rhs) == 1:  # has scalar
            for tensor in term.tensors:
                for index in tensor.indices:
                    space = convert_space(index.space)
                    if index not in externals_map:
                        externals_map[index] = externals_copy[space].pop(0)

        # Get output TensorSymbol name and rank:
        out_tensors = [tensor for tensor in term.tensors if tensor.name == ""]
        base_name = return_value
        rank = sum([len(tensor.indices) for tensor in out_tensors])
        if rank > 0:
            # Get indices:
            inds = []
            for tensor in out_tensors:
                for index in tensor.indices:
                    space = convert_space(index.space)
                    if index in dummies_map:
                        inds.append(dummies_map[index])
                    elif index in externals_map:
                        inds.append(externals_map[index])
                    else:
                        raise ValueError("Shouldn't happen")

            # Get group entry:
            group = []
            if base_name not in skip_symmetry:
                for i, tensor in enumerate(out_tensors):
                    for j, (perm, sign) in enumerate(tensor.sym.tlist):
                        acc = IDENTITY if sign == 1 else NEGATIVE
                        if i == 0:
                            group.append((perm, acc))
                        else:
                            # Combine group for two sets of output indices:
                            # TODO check this - it happens when generating the ket vectors for moments
                            perm_old, acc_old = group[j]
                            perm = perm_old + [p + (max(perm_old)+1) for p in perm]
                            acc = acc_old if sign == 1 else \
                                    {IDENTITY: NEGATIVE, NEGATIVE: IDENTITY}[acc_old]
                            group[j] = (perm, acc)

            # Build TensorSymbol:
            symb = TensorSymbol(base_name, rank, group, particles[base_name])

            # Build Tensor:
            tensor = symb[tuple(inds)]
            lhs = tensor

        else:
            symb = sympy.Symbol(base_name)
            lhs = symb

        if lhs is None:
            lhs = sympy.Symbol(return_value)

        terms.append(Term(lhs, rhs))

    # Build full index dictionary:
    indices = build_indices_dict(dummies, externals)

    # Replace indices in each term to make sure they are ordered
    # canonically:
    for i, term in enumerate(terms):
        term = term.reset_dummies(indices)
        term = term.reset_externals(indices)  # NOTE this is OK yes?
        terms[i] = term

    return terms, indices


def wicked_to_sympy(expr, indices: dict, groups: dict, particles: dict, skip_symmetry=set(), return_value: str = "res"):
    """Convert from a `wicked` strings generated via `.compile("ambit")`
    to a list of `Term`s.
    """

    spaces = (OCCUPIED, VIRTUAL, BOSON)
    occs = list("ijklmnopIJKLMNOP")
    virs = list("abcdefghABCDEFGH")
    externals = {space: [] for space in spaces}
    dummies = {space: [] for space in spaces}

    # Remove any names which conflict with index names:
    # FIXME not working - just removed f manually for now
    for lst in (occs, virs):
        if return_value in lst:
            lst.remove(return_value)
        for line in expr:
            for entry in line.split()[4::2]:
                if entry.split("[")[0] in lst:
                    lst.remove(entry.split("[")[0])

    def _convert_index(ind, cls):
        space = ind[0]
        n = int(ind[1:])
        index = indices[space][n]
        if index in occs:
            return cls(occs[n], OCCUPIED)
        else:
            return cls(virs[n], VIRTUAL)

    terms = []
    for line in expr:
        output, _, factor = line.split()[:3]
        inputs = line.split()[4::2]
        inputs[-1] = inputs[-1].strip(";")

        # Get the externals
        if "[" not in output:
            ext_inds = []
        else:
            ext_inds = output.split("[")[1].strip("]").split(",")
        ext_inds = [_convert_index(ind, ExternalIndex) for ind in ext_inds]
        for ind in ext_inds:
            if ind.space == OCCUPIED and ind not in externals[OCCUPIED]:
                externals[OCCUPIED].append(ind)
            if ind.space == VIRTUAL and ind not in externals[VIRTUAL]:
                externals[VIRTUAL].append(ind)

        # Get the LHS tensor
        if len(ext_inds):
            symb = TensorSymbol(return_value, len(ext_inds), groups[return_value], particles[return_value])
            lhs = symb[tuple(ext_inds)]
        else:
            lhs = sympy.Symbol(return_value)

        # Get the RHS tensors
        rhs = [float(factor)]
        for entry in inputs:
            # Get the indices
            if "[" not in entry:
                _inds = []
                name = entry
            else:
                name, _inds = entry.split("[")
                _inds = _inds.strip("]").split(",")
            inds = []
            for ind in _inds:
                ext_ind = _convert_index(ind, ExternalIndex)
                if ext_ind in ext_inds:
                    inds.append(ext_ind)
                else:
                    inds.append(_convert_index(ind, DummyIndex))
            for ind in inds:
                if type(ind) is DummyIndex:
                    if ind.space == OCCUPIED and ind not in dummies[OCCUPIED]:
                        dummies[OCCUPIED].append(ind)
                    if ind.space == VIRTUAL and ind not in dummies[VIRTUAL]:
                        dummies[VIRTUAL].append(ind)

            # Get the tensor
            symb = TensorSymbol(name, len(inds), groups[name], particles[name])
            rhs.append(symb[tuple(inds)])

        terms.append(Term(lhs, rhs))

    # Add the rest of the indices as dummies
    for i in range(len(occs)):
        if _convert_index("o%d"%i, ExternalIndex) not in externals[OCCUPIED]:
            ind = _convert_index("o%d"%i, DummyIndex)
            if ind not in dummies[OCCUPIED]:
                dummies[OCCUPIED].append(ind)
    for i in range(len(virs)):
        if _convert_index("v%d"%i, ExternalIndex) not in externals[VIRTUAL]:
            ind = _convert_index("v%d"%i, DummyIndex)
            if ind not in dummies[VIRTUAL]:
                dummies[VIRTUAL].append(ind)

    # Build full index dictionary:
    indices = build_indices_dict(dummies, externals)

    # Replace indices in each term to make sure they are ordered
    # canonically:
    for i, term in enumerate(terms):
        term = term.reset_dummies(indices)
        term = term.reset_externals(indices)  # NOTE this is OK yes?
        terms[i] = term

    return terms, indices


def pdaggerq_to_sympy(terms, groups: dict, particles: dict, return_value: str = "res", return_indices: tuple = None):
    """Convert from a list `pdagger` contracted strings to a list
    of `Term`s.
    """

    spaces = (OCCUPIED, VIRTUAL, BOSON)
    occs = list("ijklmnopIJKLMNO")
    virs = list("abcdeghABCDEFGH")
    externals = {space: [] for space in spaces}
    dummies = {space: [] for space in spaces}

    # Remove any names which conflict with index names:
    # TODO - just removed f, P manually for now

    from pdaggerq.config import OCC_INDICES, VIRT_INDICES
    def _convert_index(ind, cls):
        if ind in OCC_INDICES:
            n = OCC_INDICES.index(ind)
            return cls(occs[n], OCCUPIED)
        else:
            n = VIRT_INDICES.index(ind)
            return cls(virs[n], VIRTUAL)

    # Dissolve permutation operators
    i = 0
    while i != len(terms):
        perm_ops = [t for t in terms[i] if t.startswith("P")]
        terms_i = [t for t in terms[i] if not t.startswith("P")]
        if len(perm_ops):
            swaps = []
            for perm_op in perm_ops:
                i1, i2 = perm_op.replace("P", "").replace("(", "").replace(")", "").split(",")
                swaps.append(({i2: i1, i1: i2},))
            for swap in itertools.product(*swaps):
                ind_map = {k: v for d in swap for k, v in d.items()}
                new_terms_i = [terms_i[0]]
                for j, tensor in enumerate(terms_i[1:]):
                    if tensor.startswith("<"):
                        inds = tensor.replace("<", "").replace(">", "").replace("||", ",").split(",")
                        inds = tuple(ind_map.get(ind, ind) for ind in inds)
                        new_terms_i.append("<%s,%s||%s,%s>" % inds)
                    else:
                        name = tensor.split("(")[0]
                        inds = tensor.split("(")[1].replace(")", "").split(",")
                        inds = tuple(ind_map.get(ind, ind) for ind in inds)
                        new_terms_i.append("%s(%s)" % (name, ",".join(inds)))
                terms.append(new_terms_i)
        terms[i] = terms_i
        i += 1

    # Get the externals:
    ext_inds = []
    if return_indices is not None:
        for ind in return_indices:
            ind = _convert_index(ind, ExternalIndex)
            ext_inds.append(ind)
            if ind.space == OCCUPIED:
                externals[OCCUPIED].append(ind)
            else:
                externals[VIRTUAL].append(ind)

    # Get the LHS tensor
    if len(ext_inds):
        symb = TensorSymbol(return_value, len(ext_inds), groups[return_value], particles[return_value])
        lhs = symb[tuple(ext_inds)]
    else:
        lhs = sympy.Symbol(return_value)

    sympy_terms = []
    for term in terms:
        # Get the RHS tensors
        rhs = [float(term[0].replace("+", ""))]
        for entry in term[1:]:
            # Get the indices
            if entry.startswith("<"):
                name = "v"
                _inds = entry.replace("<", "").replace(">", "").replace("||", ",").split(",")
            else:
                name = entry.split("(")[0]
                _inds = entry.split("(")[1].replace(")", "").split(",")
            inds = []
            for ind in _inds:
                ext_ind = _convert_index(ind, ExternalIndex)
                if ext_ind in ext_inds:
                    inds.append(ext_ind)
                else:
                    inds.append(_convert_index(ind, DummyIndex))
            for ind in inds:
                if type(ind) is DummyIndex:
                    if ind.space == OCCUPIED and ind not in dummies[OCCUPIED]:
                        dummies[OCCUPIED].append(ind)
                    if ind.space == VIRTUAL and ind not in dummies[VIRTUAL]:
                        dummies[VIRTUAL].append(ind)

            # Get the tensor
            symb = TensorSymbol(name, len(inds), groups[name], particles[name])
            rhs.append(symb[tuple(inds)])

        sympy_terms.append(Term(lhs, rhs))

    terms = sympy_terms

    # Add the rest of the indices as dummies
    for i in range(len(occs)):
        pq_index = OCC_INDICES[i]
        if _convert_index(pq_index, ExternalIndex) not in externals[OCCUPIED]:
            ind = _convert_index(pq_index, DummyIndex)
            if ind not in dummies[OCCUPIED]:
                dummies[OCCUPIED].append(ind)
    for i in range(len(virs)):
        pq_index = VIRT_INDICES[i]
        if _convert_index(pq_index, ExternalIndex) not in externals[VIRTUAL]:
            ind = _convert_index(pq_index, DummyIndex)
            if ind not in dummies[VIRTUAL]:
                dummies[VIRTUAL].append(ind)

    # Build full index dictionary:
    indices = build_indices_dict(dummies, externals)

    # Replace indices in each term to make sure they are ordered
    # canonically:
    for i, term in enumerate(terms):
        term = term.reset_dummies(indices)
        term = term.reset_externals(indices)  # NOTE this is OK yes?
        terms[i] = term

    return terms, indices


def sympy_to_drudge(terms, indices, dr=None, skip_symmetry=set(), restricted=True):
    """Convert from a list of `Term` to `drudge.TensorDef`.
    """

    # Initialise drudge:
    if dr is None:
        from dummy_spark import SparkContext
        ctx = SparkContext()
        dr = drudge.Drudge(ctx)

    # Declare spaces:
    nocc = sympy.Symbol("nocc")
    nvir = sympy.Symbol("nvir")
    occ = drudge.Range("occ", 0, nocc)
    vir = drudge.Range("vir", 0, nvir)
    nocca, noccb = (sympy.Symbol("nocc[0]"), sympy.Symbol("nocc[1]"))
    nvira, nvirb = (sympy.Symbol("nvir[0]"), sympy.Symbol("nvir[1]"))
    occa, occb = (drudge.Range("occ[0]", 0, nocca), drudge.Range("occ[1]", 0, noccb))
    vira, virb = (drudge.Range("vir[0]", 0, nvira), drudge.Range("vir[1]", 0, nvirb))
    nbos = sympy.Symbol("nbos")
    bos = drudge.Range("bos", 0, nbos)

    # Declare dummy indices:
    for rng, space in [(occ, OCCUPIED), (vir, VIRTUAL), (bos, BOSON)]:
        if (space, None) not in indices["dummies"]:
            continue
        dumms = [sympy.Symbol(repr(index)) for index in indices["dummies"][space, None]]
        dr.set_dumms(rng, dumms)
    for rng, space in [(occa, OCCUPIED), (vira, VIRTUAL)]:
        if (space, ALPHA) not in indices["dummies"]:
            continue
        dumms = [sympy.Symbol(repr(index)) for index in indices["dummies"][space, ALPHA]]
        dr.set_dumms(rng, dumms)
    for rng, space in [(occb, OCCUPIED), (virb, VIRTUAL)]:
        if (space, BETA) not in indices["dummies"]:
            continue
        dumms = [sympy.Symbol(repr(index)) for index in indices["dummies"][space, BETA]]
        dr.set_dumms(rng, dumms)
    dr.add_resolver_for_dumms()

    # Build einstein summations:
    expr = 0
    groups = {}
    for term in terms:
        part = term.factor
        for tensor in term.rhs_tensors:
            base = sympy.IndexedBase(tensor.base.name)
            indices = [sympy.Symbol(repr(index)) for index in tensor.indices]
            part *= base[tuple(indices)]
            if base not in groups:
                groups[base] = tensor.group
        expr += part

    # Build TensorDef:
    assert all(terms[0].lhs == term.lhs for term in terms)
    if isinstance(terms[0].lhs, Tensor):
        lhs = [sympy.IndexedBase(terms[0].lhs.base.name)]
        if all(i.spin is None for i in terms[0].lhs.indices):
            for index in terms[0].lhs.indices:
                lhs.append((
                    sympy.Symbol(repr(index)),
                    {OCCUPIED: occ, VIRTUAL: vir, BOSON: bos}[index.space],
                ))
        else:
            spins = [i.spin for i in terms[0].lhs.indices]
            for index, spin in zip(terms[0].lhs.indices, spins):
                lhs.append((
                    sympy.Symbol(repr(index)),
                    {
                        OCCUPIED: (occa if spin == ALPHA else occb),
                        VIRTUAL: (vira if spin == ALPHA else virb),
                        BOSON: bos,
                    }[index.space],
                ))
    else:
        lhs = [sympy.Symbol(terms[0].lhs.name)]
    rhs = dr.einst(expr)
    tensordef = dr.define(*lhs, rhs)

    # Set symmetry:
    for symb, group in groups.items():
        if symb.name in skip_symmetry:
            continue
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
    # FIXME this messes up indices
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
