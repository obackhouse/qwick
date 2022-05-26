"""Perform spin integration on an expression in the format of the
`drudge` program.
"""

from typing import List, Tuple, Dict, Union
import drudge
import sympy


"""
PermType: 
    Dictionary which has keys (symbol, valence) and has items which
    are a tuple of permutations (tuples of int) and an action.
"""

PermType = Dict[Tuple[sympy.IndexedBase, int], Tuple[Tuple[int], int]],


"""
ParticleList:
    Dictionary which has keys (symbol, valence) and has items
    (particle_type, particle_id).
"""

ParticleList = Dict[Tuple[sympy.IndexedBase, int], Tuple[int, int]]


def action_to_callable(acc):
    """Convert a `drudge` action to a callable which appropriately
    transforms the argument.
    """

    if acc == drudge.IDENT:
        out = lambda x: x
    elif acc == drudge.NEG:
        out = lambda x: -x
    elif acc == drudge.CONJ:
        out = lambda x: sympy.functions.elementary.conjugate(x)

    return out


def get_permuting_function(perms, atom=sympy.Indexed):
    """Get a function which permutes the indices of all `symp.atoms`
    of type `atom` according the to permutations `perms`.
    """

    def func(term):
        repl = {}
        for tensor in term.amp.atoms(atom):
            new_tensor = 0
            for perm, acc in perms[(tensor.base, len(tensor.indices))]:
                permuted_indices = tuple(tensor.indices[i] for i in perm)
                new_tensor += action_to_callable(acc)(tensor.base[permuted_indices])
            repl[tensor] = new_tensor
        return term.subst(repl)

    return func


def postprocess_restricted(expr):
    """Post-processing for restricted spatial orbitals.
    """

    def func(term):
        repl = {}
        for tensor in term.amp.atoms(sympy.Indexed):
            spatial_indices = tuple(index[0] for index in tensor.indices)
            new_tensor = tensor.base[spatial_indices]
            repl[tensor] = new_tensor
        return term.subst(repl)

    expr = expr.map(func)
    expr = expr.simplify()

    return ["{res}"], [expr]


def default_add_spin_labels(base_name, spins):
    """Add spin labels to a symbol name.
    """

    name = base_name + "_"

    for spin in spins:
        if spin == 0:
            name += "α"
        elif spin == 1:
            name += "β"
        else:
            raise ValueError(
                    "Default `add_spin_labels` function does not "
                    "support spin index %d, must pass a custom "
                    "function for `unrestricted=True`. " % spin,
            )

    return name


def postprocess_unrestricted(expr, add_spin_labels: callable = default_add_spin_labels):
    """Post-processing for unrestricted spatial orbitals.
    """

    def _get_output_spins(term):
        indices = {}
        for tensor in term.amp.atoms(sympy.Indexed):
            for index, spin in tensor.indices:
                indices[index] = spin
        for index, range_ in term.sums:
            if index in indices:
                del indices[index]
        return indices

    # Find all possible output index permutations:
    outputs = set()
    def func(term):
        indices = _get_output_spins(term)
        outputs.add(tuple(indices.items()))
    expr.map(func)
    outputs = sorted(outputs, key=lambda x: sum(y[1] for y in x))

    # For each output, generate a new expression:
    exprs = []
    output_templates = []
    for output in outputs:
        def func(term):
            indices = _get_output_spins(term)
            return tuple(indices.items()) == output
        spins = [x[1] for x in output]
        exprs.append(expr.filter(func))
        output_templates.append(add_spin_labels("{res}", spins))

    # For each expression, generate new tensors with spin labels:
    for i, expr in enumerate(exprs):
        def func(term):
            repl = {}
            for tensor in term.amp.atoms(sympy.Indexed):
                indices = tuple(index[0] for index in tensor.indices)
                spins = tuple(index[1] for index in tensor.indices)
                base = sympy.IndexedBase(add_spin_labels(tensor.base.name, spins))
                repl[tensor] = base[indices]
            return term.subst(repl)
        expr = expr.map(func)
        expr = expr.simplify()
        exprs[i] = expr

    return output_templates, exprs


def set_symmetry(dr: drudge.Drudge, perms: PermType):
    """Set the symmetries using a `PermType`.
    """

    def is_identity(perm):
        return all(i == j for i, j in enumerate(perm))

    for (symbol, valence), perm in perms.items():
        if perm is None:
            perm = []

        perm = [p for p in perm if not is_identity(p[0])]
        perm = [drudge.Perm(p, f) for p, f in perm]
        if not len(perm):
            perm = [None]

        dr.set_symm(
                symbol,
                *perm,
                valence=valence,
        )


def permutations_with_signs(seq):
    """Generate permutations of seq, yielding also a sign which is
    equal to +1 for an even number of swaps, and -1 for an odd number
    of swaps.
    """

    def _permutations(seq):
        if not seq:
            return [[]]

        items = []
        for i, item in enumerate(_permutations(seq[:-1])):
            inds = range(len(item) + 1)
            if i % 2 == 0:
                inds = reversed(inds)
            items += [item[:i] + seq[-1:] + item[i:] for i in inds]

        return items

    return [(item, -1 if i % 2 else 1) for i, item in enumerate(_permutations(list(seq)))]


def get_bare_groups(groups: PermType, particles: ParticleList):
    """Compute the bare groups which do not exchange particles from
    the groups which do, and the particle type and indices of each
    index in the tensor.
    """
    # TODO verify for other examples!

    def relabel(lst):
        cache = {}
        n = 0
        for i in lst:
            if i not in cache:
                cache[i] = n
                n += 1
        return [cache[i] for i in lst]

    bare_groups = {}

    for (symbol, valence), perms in groups.items():
        bare_perms = []
        for perm in perms:
            particles_permuted = [particles[symbol, valence][i] for i in perm[0]]
            if tuple(relabel(particles[symbol, valence])) == tuple(relabel(particles_permuted)):
                bare_perms.append(perm)

        bare_groups[symbol, valence] = bare_perms

    return bare_groups


def get_spin_perms(groups: PermType, particles: ParticleList):
    """Compute the spin permutations from the particle type and indices
    of each index in the tensor.
    """
    # TODO verify for other examples!

    def replace(lst, pattern, replacement):
        out = lst.copy()
        for i in range(len(out)):
            if out[i] == pattern:
                out[i] = replacement
        return out

    spin_perms = {}

    for (symbol, valence), perms in groups.items():
        spin_perm = [particles[symbol, valence].copy()]
        # Each fermion gets [0, 1]
        # Each boson gets [2, 3]
        # i.e. [(drudge.FERMI, 0), (drudge.FERMI, 1), (drudge.FERMI, 0), (drudge.FERMI, 1)]
        particle_types = set(particles[symbol, valence])
        for particle in particle_types:
            if particle[0] == drudge.FERMI:
                spin_perm = [replace(p, particle, s) for p in spin_perm for s in (0, 1)]
            else:
                spin_perm = [replace(p, particle, s) for p in spin_perm for s in (2, 3)]

        spin_perms[symbol, valence] = [(s, drudge.IDENT) for s in spin_perm]

    return spin_perms


def get_symmetry_perms(groups: PermType, particles: ParticleList):
    """Compute the symmetry permutations from the particle type and
    indices of each index in the tensor. These permutations should
    dissolve the bosonic symmetry and/or fermionic antisymmetry with
    particle exchange.
    """
    # TODO verify for other examples!

    symmetry_perms = {}

    for (symbol, valence), perms in groups.items():
        identity = list(range(valence))
        symmetry_perms[symbol, valence] = []

        # Each particle should have no more than 2 indices - antisymmetrise
        # on the final index of each particle.
        particle_counts = {}
        first_index = {}
        for i, particle in enumerate(particles[symbol, valence]):
            particle_counts[particle, valence] = particle_counts.get(particle, 0) + 1
            if particle not in first_index:
                first_index[particle] = i
        assert all(n <= 2 for n in particle_counts.values())

        # Resulting tensor needs to be antisymmetric wrt fermion exchange
        # and symmetric wrt boson exchange. First fermions:
        fermion_indices = [val for key, val in first_index.items() if key[0] == drudge.FERMI]
        for new_fermion_indices, sign in permutations_with_signs(fermion_indices):
            perm = list(range(valence))
            for start, end in zip(fermion_indices, new_fermion_indices):
                perm[start] = identity[end]
            symmetry_perms[symbol, valence].append((perm, drudge.IDENT if sign == 1 else drudge.NEG))

        # Next, transform the antisymmetric fermion permutations to also
        # give symmetry with exchange of bosonic indices:
        boson_indices = [val for key, val in first_index.items() if key[0] == drudge.BOSE]
        for new_boson_indices, sign in permutations_with_signs(boson_indices):
            perms = symmetry_perms[symbol, valence].copy()
            symmetry_perms[symbol, valence] = []
            for i, (perm, sign) in enumerate(perms):
                for start, end in zip(boson_indices, new_boson_indices):
                    perm[start] = identity[end]
                symmetry_perms[symbol, valence].append((perm, sign))  # No sign change

        symmetry_perms[symbol, valence] = tuple(symmetry_perms[symbol, valence])

    return symmetry_perms


def spin_integrate(
        dr: drudge.Drudge,
        tensordef: drudge.TensorDef,
        groups: PermType,
        particles: ParticleList,
        restricted: bool = True,
        add_spin_labels: callable = default_add_spin_labels,
):
    """Perform spin integration on an expression.

    Arguments
    ---------
    dr: drudge.Drudge
        `drudge` object.
    tensordef: drudge.TensorDef
        `drudge` expression to perform spin-integration on.
    groups: PermType
        Maps indices to symmetry-equivalent permutations with a given
        action for each `sympy.IndexedBase` with the physically
        correct symmetry relations with respect to particle exchange
        (antisymmetry for fermions, symmetry for bosons).
    particles: ParticleList
        Dictionary giving the type and id of each particle in the
        tensor for each `sympy.IndexedBase`.
    restricted: bool, optional
        If True, sum over final spin indices to return spin-free
        expressions with spin symmetry. Default value is True.
    add_spin_labels: callable, optional
        Function which takes the name of a `sympy.IndexedBase` and a
        list of integer spin labels for each index and returns a new
        name for the spin-labelled `sympy.IndexedBase`. Only valid
        for `restricted=False`. Default value is the function
        `default_add_spin_labels`.

    Returns
    -------
    output_templates: list of str
        Templates for the string format of the output tensors, where
        {res} is formatted with the original string. Allows decoration
        of the original output tensors for i.e. unrestricted cases.
    exprs: list of drudge.Tensor
        `drudge` expressions for spin-integrated expressions.
    """

    definition = tensordef.lhs
    expr = tensordef.rhs

    # Generate additional symmetry information:
    print("Generating groups", flush=True)
    bare_groups = get_bare_groups(groups, particles)
    spin_perms = get_spin_perms(groups, particles)
    symmetry_perms = get_symmetry_perms(groups, particles)

    # Set the symmetry of each symbol to the antisymmetric groups:
    print("Applying full symmetry groups", flush=True)
    set_symmetry(dr, groups)
    expr = expr.simplify()
    print(" -> n_terms = %d" % expr.n_terms, flush=True)

    # Remove symmetry before dissolving antisymmetry:
    set_symmetry(dr, {key: None for key in groups.keys()})

    # Dissolve the antisymmetry:
    print("Dissolving antisymmetry for fermions and symmetry for bosons", flush=True)
    func = get_permuting_function(symmetry_perms)
    expr = expr.map(func)
    #expr = expr.simplify()
    print(" -> n_terms = %d" % expr.n_terms, flush=True)

    # Declare symmetry of each symbol to the symmetric groups:
    print("Applying bare symmetry groups", flush=True)
    set_symmetry(dr, bare_groups)
    expr = expr.simplify()
    print(" -> n_terms = %d" % expr.n_terms, flush=True)

    # Dissolve the spin orbitals in the tensors
    # NOTE: we don't actually convert the summed indices at the moment,
    # but it shouldn't matter in the final expressions I don't think.
    print("Dissolving the spin orbitals", flush=True)
    def func(term):
        repl = {}
        # Only want to get non-zero contributions
        for tensor in term.amp.atoms(sympy.Indexed):
            new_tensor = 0
            for perm, acc in spin_perms[tensor.base, len(tensor.indices)]:
                spin_indices = tuple((index, i) for index, i in zip(tensor.indices, perm))
                new_tensor += action_to_callable(acc)(tensor.base[spin_indices])
            repl[tensor] = new_tensor
        return term.subst(repl)
    expr = expr.map(func)
    expr = expr.simplify()
    print(" -> n_terms = %d" % expr.n_terms, flush=True)

    # In each contraction, use the particle numbers of each index to
    # determine if the contraction is zero due to spin:
    # TODO this can almost certainly be done in O(n)
    # TODO can we combine with the dissolution of the spin-orbitals for efficiency?
    print("Identifying spin forbidden terms", flush=True)
    def func(term):
        tensors = list(term.amp.atoms(sympy.Indexed))
        for i, t1 in enumerate(tensors):
            s1 = {i: s for i, s in t1.indices}
            for t2 in tensors[:i]:
                s2 = {i: s for i, s in t2.indices}
                intersection = s1.keys() & s2.keys()
                for key in intersection:
                    if s1[key] != s2[key]:
                        return False
        return True
    expr = expr.filter(func)
    expr = expr.simplify()
    print(" -> n_terms = %d" % expr.n_terms, flush=True)

    # If a restricted expression is required, sum over spin, otherwise
    # move spin indices and sum into separate tensors.
    print("Casting to %sHF expressions" % ("R" if restricted else "U"), flush=True)
    if restricted:
        templates, exprs = postprocess_restricted(expr)
    else:
        templates, exprs = postprocess_unrestricted(expr, add_spin_labels=default_add_spin_labels)
    print(" -> n_terms = %d" % expr.n_terms, flush=True)

    # Rebuild the tensor definitions:
    eqns = []
    for template, expr in zip(templates, exprs):
        # TODO generalise?
        if isinstance(definition, sympy.Indexed):
            base = sympy.IndexedBase(template.format(res=definition.base))
            newdefinition = base[tuple(definition.indices)]
        elif isinstance(definition, sympy.Symbol):
            newdefinition = template.format(res=definition.name)
        else:
            raise NotImplementedError

        expr = expr.sort()
        tensor = dr.define(newdefinition, expr)
        eqns.append(tensor)

    return eqns


if __name__ == "__main__":
    from dummy_spark import SparkContext

    ctx = SparkContext()
    dr = drudge.Drudge(ctx)

    # Declare variables:
    v = sympy.IndexedBase("v")

    # Declare space ranges:
    nocc = sympy.Symbol("nocc")
    nvir = sympy.Symbol("nvir")
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
            (v, 4): (
                ([0, 1, 2, 3], drudge.IDENT),
                ([1, 0, 3, 2], drudge.IDENT),
                ([1, 0, 2, 3], drudge.NEG),
                ([0, 1, 3, 2], drudge.NEG),
            ),
    }
    particles = {
            (v, 4): [
                (drudge.FERMI, 0),
                (drudge.FERMI, 1),
                (drudge.FERMI, 0),
                (drudge.FERMI, 1),
            ],
    }
    # Now derived:
    #bare_groups = {
    #        v: (
    #            ([0, 1, 2, 3], drudge.IDENT),
    #            ([1, 0, 3, 2], drudge.IDENT),
    #        ),
    #}
    # Now derived:
    #symmetry_perms = {
    #        v: (
    #            ([0, 1, 2, 3], drudge.IDENT),
    #            ([0, 1, 3, 2], drudge.NEG),
    #        ),
    #}
    # Now derived:
    #spin_perms = {
    #        v: (
    #            ([0, 0, 0, 0], drudge.IDENT),
    #            ([0, 1, 0, 1], drudge.IDENT),
    #            ([1, 0, 1, 0], drudge.IDENT),
    #            ([1, 1, 1, 1], drudge.IDENT),
    #        ),
    #}

    # Write expressions:
    expr = dr.einst(0.25 * v[i, j, a, b] * v[a, b, i, j])
    #expr = dr.einst(0.25 * v[k, i, c, a] * v[l, i, d, a])
    tensor = dr.define(sympy.Symbol("e"), expr)

    # Spin integrate:
    eqns = spin_integrate(
            dr,
            tensor,
            groups,
            particles,
            restricted=True,
    )

    for eqn in eqns:
        print(eqn, "\n")
