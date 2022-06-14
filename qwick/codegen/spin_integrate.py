"""
Class definitions.
"""

import sympy
from sympy import S
from sympy.core.assumptions import StdFactKB
from sympy.core.logic import fuzzy_bool
from sympy.core.cache import cacheit
import copy
from numbers import Number
from collections.abc import Sequence
from typing import Tuple, List, Union, Type


SpaceType = int
ParticlesType = List[Tuple[int, int]]
GroupType = List[Tuple[Tuple[int], int]]
SpinType = int

def singleton_factory(identifier, name):
    class Singleton(int):
        def __repr__(self):
            return name
    return Singleton(identifier)

OCCUPIED = singleton_factory(0, "occupied")
VIRTUAL = singleton_factory(1, "virtual")
BOSON = SBOSON = SCALAR_BOSON = singleton_factory(2, "scalar-boson")
VBOSON = VECTOR_BOSON = singleton_factory(3, "vector-boson")
FERMION = singleton_factory(4, "fermion")
IDENTITY = singleton_factory(5, "identity")
NEGATIVE = singleton_factory(6, "negative")
CONJUGATE = singleton_factory(7, "conjugate")
ALPHA = singleton_factory(8, "alpha")
BETA = singleton_factory(9, "beta")


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


def action_to_callable(action):
    if action == IDENTITY:
        return lambda x: x
    elif action == NEGATIVE:
        return lambda x: -x
    elif action == CONJUGATE:
        return lambda x: sympy.conjugate(x)
    else:
        raise ValueError("action = %s", action)


def expand_products(expr):
    """Return a list of products from an expression without any
    explicit `sympy.Add` objects.
    """

    expr = expr.expand()

    if isinstance(expr, sympy.Add):
        return expr.args
    else:
        return [expr]


class AIndex(sympy.Symbol):
    """Abstract base class for an Index.
    """

    __slots__ = ("name", "space", "spin")

    name: str
    space: SpaceType
    spin: SpinType

    def __new__(cls, name: str, space: SpaceType, spin: SpinType = None, **assumptions):
        cls._sanitize(assumptions, cls)
        if "," in name or ":" in name:
            symbols = sympy.symbols(name, cls=lambda name: cls(name, space, spin=spin))
            return symbols
        return AIndex.__xnew_cached__(cls, name, space, spin, **assumptions)

    @staticmethod
    def __xnew__(cls, name, space, spin, **assumptions):
        obj = sympy.Expr.__new__(cls)
        obj.name = name
        obj.space = space
        obj.spin = spin
        tmp_asm_copy = assumptions.copy()
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        assumptions['commutative'] = is_commutative
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy
        return obj

    @staticmethod
    @cacheit
    def __xnew_cached__(cls, name, space, spin, **assumptions):
        return AIndex.__xnew__(cls, name, space, spin, **assumptions)

    def __getnewargs_ex__(self):
        return ((self.name, self.space, self.spin), self.assumptions0)

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def __hash__(self):
        return hash((self.name, self.space, self.spin))

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name, self.space, self.spin)), S.One.sort_key(), S.One

    def __str__(self):
        out = sympy.Symbol.__str__(self)
        if self.spin != None:
            if self.spin == ALPHA:
                out += "α"
            elif self.spin == BETA:
                out += "β"
            else:
                raise ValueError
        return out

    def __repr__(self):
        return str(self)


class ExternalIndex(AIndex):
    """Class for an external Index.
    """

    pass


class DummyIndex(AIndex):
    """Class for a dummy Index.
    """

    pass


class TensorSymbol(sympy.IndexedBase):
    """Class for a single Tensor symbol instance.
    """

    def __new__(self, name: str, rank: int, group: GroupType, particles: ParticlesType):
        base = sympy.IndexedBase.__new__(TensorSymbol, name)
        base.rank = rank
        base.group = group
        base.particles = particles
        base.bare_group = base.get_bare_group()
        base.exchange_group = base.get_exchange_group()
        base.spin_group = base.get_spin_group()
        return base

    def get_bare_group(self):
        """Use `group` and `particles` to produce a new `GroupType`
        corresponding to the permutations of the bare tensor after
        symmetrisation/antisymmetrisation has been expanded.
        """

        def relabel(lst):
            cache = {}
            n = 0
            for i in lst:
                if i not in cache:
                    cache[i] = n
                    n += 1
            return [cache[i] for i in lst]

        # Find groups which do not permute indices between particles:
        group = []
        for perm, acc in self.group:
            particles_permuted = [self.particles[i] for i in perm]
            if tuple(relabel(self.particles)) == tuple(relabel(particles_permuted)):
                group.append((perm, acc))

        return group

    def get_exchange_group(self):
        """Use `group` and `particles` to produce a new `GroupType`
        corresponding to the permutations of the bare tensor which
        expand the symmetrisation/antisymmetrisation of the original
        tensor.
        """

        identity = list(range(self.rank))
        group = []

        # Each particle should have no more than 2 indices - symmetrise
        # on the final index of each particle:
        particle_counts = {}
        first_index = {}
        for i, particle in enumerate(self.particles):
            particle_counts[particle] = particle_counts.get(particle, 0) + 1
            if particle not in first_index:
                first_index[particle] = i
        assert all(n <= 2 for n in particle_counts.values())

        # Resulting tensor needs to be antisymmetric wrt fermion exchange
        # and symmetry wrt boson exchange. First fermions:
        fermion_indices = [
                val for key, val in first_index.items() 
                if key[0] in (OCCUPIED, VIRTUAL, FERMION)
        ]
        for new_fermion_indices, sign in permutations_with_signs(fermion_indices):
            perm = list(range(self.rank))
            for start, end in zip(fermion_indices, new_fermion_indices):
                perm[start] = identity[end]
            group.append((perm, IDENTITY if sign == 1 else NEGATIVE))

        # Next, transform the antisymmetric fermion permutations to also
        # give symmetry with exchange of bosonic indices:
        boson_indices = [
                val for key, val in first_index.items()
                if key[0] == BOSON
        ]
        for new_boson_indices, sign in permutations_with_signs(boson_indices):
            perms = group.copy()
            group = []
            for i, (perm, sign) in enumerate(perms):
                for start, end in zip(boson_indices, new_boson_indices):
                    perm[start] = identity[end]
                group.append((perm, sign))

        return group

    def get_spin_group(self):
        """Use `group` and `particles` to produce a new `GroupType`
        corresponding to the permutations of the bare tensor which
        expand the spin orbitals of the original tensor to
        spin-labelled spatial orbitals.
        """

        def replace(lst, pattern, replacement):
            out = lst.copy()
            for i in range(len(out)):
                if out[i] == pattern:
                    out[i] = replacement
            return out

        perms = [list(self.particles).copy()]

        particle_types = set(self.particles)
        for particle in particle_types:
            if particle[0] == SCALAR_BOSON:
                spins = (None,)
            else:
                spins = (ALPHA, BETA)
            perms = [replace(p, particle, s) for p in perms for s in spins]

        perms = [(s, IDENTITY) for s in perms]

        return perms

    def __getitem__(self, indices: Tuple[AIndex], **kwargs):
        assert len(indices) == self.rank
        if sympy.tensor.indexed.is_sequence(indices):
            if self.shape and len(self.shape) != len(indices):
                raise sympy.tensor.indexed.IndexException("Rank mismatch.")
            return Tensor(self, *indices, **kwargs)
        else:
            if self.shape and len(self.shape) != 1:
                raise sympy.tensor.indexed.IndexException("Rank mismatch.")
            return Tensor(self, indices, **kwargs)

    def copy(self, name=None, rank=None, group=None, particles=None):
        if name is None:
            name = self.name
        if rank is None:
            rank = self.rank
        if group is None:
            group = self.group
        if particles is None:
            particles = self.particles
        return self.__class__(name, rank, group=group, particles=particles)


class Tensor(sympy.Indexed):
    """Class for a single Tensor instance.
    """

    def __new__(self, *args, **kwargs):
        tensor = sympy.Indexed.__new__(self, *args, **kwargs)
        tensor.particle_exchange_symmetry = True
        return tensor

    @property
    def group(self):
        if self.particle_exchange_symmetry:
            return self.base.group
        else:
            return self.base.bare_group

    @property
    def particles(self):
        return self.base.particles

    @property
    def exchange_group(self):
        return self.base.exchange_group

    @property
    def spin_group(self):
        return self.base.spin_group

    @property
    def dummy_indices(self):
        indices = []
        for index in self.indices:
            # FIXME add spin labels to AIndex classes
            if isinstance(index, DummyIndex) or (
                    isinstance(index, (Sequence, sympy.Tuple)) and isinstance(index[0], DummyIndex)):  # FIXME
                indices.append(index)
        return indices

    @property
    def external_indices(self):
        indices = []
        for index in self.indices:
            # FIXME add spin labels to AIndex classes
            if isinstance(index, ExternalIndex) or (
                    isinstance(index, (Sequence, sympy.Tuple)) and isinstance(index[0], ExternalIndex)):  # FIXME
                indices.append(index)
        return indices

    #def _reset_indices(self, indices, indices_possible):
    #    """Reset indices for canonicalization.
    #    """
    #    raise NotImplementedError  # Not used: move from Term to here, FIXME

    #    indices_possible = indices_possible.copy()
    #    indices_possible_inv = {}
    #    for key, val in indices_possible.items():
    #        for v in val:
    #            indices_possible_inv[v] = key

    #    subs = {}
    #    for index in indices:
    #        new_index = indices_possible[indices_possible_inv[index]].pop(0)
    #        subs[index] = new_index

    #    return self.xreplace(subs)

    #def reset_dummies(self, indices):
    #    """Reset the dummies for canonicalization.
    #    """

    #    return self._reset_indices(self.dummy_indices, indices)

    #def reset_externals(self, indices):
    #    """Reset the externals for canonicalization.
    #    """

    #    return self._reset_indices(self.external_indices, indices)

    def canonicalize(self, indices=None):
        """Canonicalize the indices of the Tensor according to the
        permitted permutations. If `dummies` and `externals` are
        passed, also canonicalize the index names.
        """

        if len(self.group) == 0:
            return self

        group = self.group
        best = {"indices": None, "action": None}

        for perm, action in group:
            indices = tuple(self.indices[i] for i in perm)
            if best["indices"] is None or indices < best["indices"]:
                best["indices"] = indices
                best["action"] = action

        return action_to_callable(best["action"])(self.copy(indices=best["indices"]))

    def copy(self, base: Type = None, indices: List = None):
        """Return a copy with different indices.
        """

        if indices is None:
            indices = copy.copy(self.indices)
        if base is None:
            base = self.base.copy()

        tensor = base[indices]
        tensor.particle_exchange_symmetry = self.particle_exchange_symmetry

        return tensor

    def permute_indices(self, perm: Tuple[int]):
        """Return a Tensor with indices permuted.
        """

        permuted_indices = [self.indices[i] for i in perm]
        permuted_indices = tuple(permuted_indices)

        return self.copy(indices=permuted_indices)

    def __lt__(self, other):
        if isinstance(other, Number):
            return True
        else:
            return (self.base.name, self.indices) < (other.base.name, other.indices)

    def __eq__(self, other):
        return (self.base, self.indices) == (other.base, other.indices)

    def __str__(self):
        return "%s_{%s}" % (self.base.name, ", ".join([str(i) for i in self.indices]))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.base, self.indices, self.particle_exchange_symmetry))


class Term:
    """Class for a single Term instance, with a Tensor on the LHS and
    a list of Tensors and scalars on the RHS.
    """

    def __init__(self, lhs: Tensor, rhs: List[Union[sympy.Number, Tensor]]):
        self.parent = None
        self.lhs = lhs

        if len(rhs) == 0:
            raise ValueError("Empty RHS!")
        elif len(rhs) == 1:
            self.rhs = rhs
        else:
            mul = sympy.Mul(*rhs).expand()
            if isinstance(mul, sympy.Add):
                raise NotImplementedError
            elif isinstance(mul, Tensor):
                self.rhs = (mul,)
            else:
                self.rhs = tuple(mul.args)

        assert isinstance(self.lhs, (sympy.Symbol, Tensor))
        assert all(isinstance(x, (sympy.Number, Number, Tensor)) for x in self.rhs)

    @property
    def rhs_tensors(self):
        for arg in self.rhs:
            if isinstance(arg, Tensor):
                yield arg

    @property
    def all_tensors(self):
        if isinstance(self.lhs, sympy.Indexed):
            yield self.lhs
        for tensor in self.rhs_tensors:
            yield tensor

    @property
    def factor(self):
        for arg in self.rhs:
            if isinstance(arg, Number):
                return arg
        return sympy.S.One

    def _get_indices(self, tensors, dummy=True):
        done = set()
        indices = []
        for tensor in tensors:
            for index in (tensor.dummy_indices if dummy else tensor.external_indices):
                if index not in done:
                    indices.append(index)
                    done.add(index)
        return indices

    @property
    def dummy_indices(self):
        return self._get_indices(self.all_tensors, True)

    @property
    def external_indices(self):
        return self._get_indices(self.all_tensors, False)

    @property
    def indices(self):
        return self.dummy_indices + self.external_indices

    @property
    def rhs_dummy_indices(self):
        return self._get_indices(self.rhs_tensors, True)

    @property
    def rhs_external_indices(self):
        return self._get_indices(self.rhs_tensors, False)

    @property
    def rhs_indices(self):
        return self.rhs_dummy_indices + self.rhs_external_indices

    def reset_externals(self, indices):
        """Reset external indices. This should be done with care.
        """

        indices = {key: val.copy() for key, val in indices["externals"].items()}

        subs = {}
        for index in self.external_indices:
            if index not in subs:
                # Get next index in the current spin channel:
                new_index = indices[index.space, index.spin].pop(0)

                # Also pop index with same name from other spin channels:
                other_spins = {None, ALPHA, BETA}
                other_spins.remove(index.spin)
                for spin in other_spins:
                    tmp = index.__class__(index.name, index.space, spin=spin)
                    indices[index.space, spin].pop(0)

                # Assign the substitution:
                subs[index] = new_index

        # Function to substitute indices where necessary:
        def sub_indices(indices):
            new_indices = []
            for index in indices:
                if index in subs:
                    new_indices.append(subs[index])
                else:
                    new_indices.append(index)
            return tuple(new_indices)

        # Build new term:
        lhs = self.lhs
        if isinstance(lhs, sympy.Indexed):
            lhs = lhs.copy(indices=sub_indices(lhs.indices))
        rhs = [self.factor]
        for tensor in self.rhs_tensors:
            rhs.append(tensor.copy(indices=sub_indices(tensor.indices)))

        return Term(lhs, rhs)

    def reset_dummies(self, indices):
        """Reset indices for canonicalization.
        """

        indices = {key: val.copy() for key, val in indices["dummies"].items()}

        subs = {}
        for index in self.dummy_indices:
            if index not in subs:
                # Get next index in the current spin channel:
                new_index = indices[index.space, index.spin].pop(0)

                # Also pop index with same name from other spin channels:
                other_spins = {None, ALPHA, BETA}
                other_spins.remove(index.spin)
                for spin in other_spins:
                    tmp = index.__class__(index.name, index.space, spin=spin)
                    indices[index.space, spin].pop(0)

                # Assign the substitution:
                subs[index] = new_index

        # Function to substitute indices where necessary:
        def sub_indices(indices):
            new_indices = []
            for index in indices:
                if index in subs:
                    new_indices.append(subs[index])
                else:
                    new_indices.append(index)
            return tuple(new_indices)

        # Build new term:
        lhs = self.lhs
        if isinstance(lhs, sympy.Indexed):
            lhs = lhs.copy(indices=sub_indices(lhs.indices))
        rhs = [self.factor]
        for tensor in self.rhs_tensors:
            rhs.append(tensor.copy(indices=sub_indices(tensor.indices)))

        return Term(lhs, rhs)

    # FIXME this isn't doing a full canonicalisation
    def canonicalize(self, indices=None):
        """Canonicalize the indices of the Tensor according to the
        permitted permutations. If `dummies` and `externals` are
        passed, also canonicalize the index names.
        """

        lhs = self.lhs
        if isinstance(lhs, sympy.Indexed):
            lhs = lhs.canonicalize()
        rhs = [self.factor]
        for tensor in self.rhs_tensors:
            rhs.append(tensor.canonicalize())

        # Build the Term early to trigger any conversions:
        term = Term(lhs, rhs)

        lhs = term.lhs
        rhs = [term.factor] + sorted(term.rhs_tensors)
        term = Term(lhs, rhs)

        if indices is not None:
            term = term.reset_dummies(indices)

        return term

    def get_index_particles(self):
        """Get a dictionary mapping each index to a particle.
        """

        indices = set(self.dummy_indices + self.external_indices)
        index_particles = {}
        next_particle = 0

        for tensor in self.all_tensors:
            for index, particle in zip(tensor.indices, tensor.particles):
                if index in index_particles:
                    continue
                else:
                    for index1, particle1 in zip(tensor.indices, tensor.particles):
                        if index == index1:
                            continue
                        if particle == particle1 and index1 in index_particles:
                            index_particles[index] = index_particles[index1]
                            break
                    else:
                        index_particles[index] = next_particle
                        next_particle += 1

        return index_particles

    def __hash__(self):
        """NOTE: Hash does NOT include the factor.
        """

        return hash((self.lhs, tuple(self.rhs_tensors)))

    def __str__(self):
        out  = str(self.lhs)
        out += " += "

        strfac = str(float(self.factor))
        if "." in strfac:
            while strfac[-1] == "0" and strfac[-2] != ".":
                strfac = strfac[:-1]

        dummy_names = [str(index) for index in self.dummy_indices]
        out += strfac + " "
        if len(dummy_names):
            out += "sum_{%s} " % ", ".join(dummy_names)

        for term in self.rhs_tensors:
            out += str(term) + " "

        return out

    def __add__(self, other):
        if hash(self) != hash(other):
            raise ValueError(
                    "Cannot add Terms unless they are equal within "
                    "their factors."
            )

        lhs = self.lhs
        rhs = [self.factor + other.factor,] + list(self.rhs_tensors)

        return Term(lhs, rhs)

    def expand_particle_exchange_symmetry(self):
        """Remove the particle exchange symmetry by expanding through
        the permutations with their signs for each term. Returns a
        list of terms.
        """
        # TODO move to Tensor class?

        expanded_tensors = []
        for tensor in self.rhs_tensors:
            new_tensor = 0
            for perm, action in tensor.exchange_group:
                new_tensor_contrib = tensor.permute_indices(perm)
                new_tensor_contrib = action_to_callable(action)(new_tensor_contrib)
                new_tensor += new_tensor_contrib
            expanded_tensors.append(new_tensor)

        # Expand the product between expanded_tensors
        expanded_tensors = expand_products(sympy.Mul(*expanded_tensors))

        terms = []
        for product in expanded_tensors:
            if isinstance(product, Tensor):
                args = (self.factor, product)
            else:
                args = (self.factor, *product.args)
            new_term = Term(self.lhs, args)
            if new_term.factor != 0:
                terms.append(new_term)

        # Set the particle exchange symmetry flag of the new tensors:
        for term in terms:
            for tensor in term.all_tensors:
                tensor.particle_exchange_symmetry = False

        # Set the parent of each of the new terms:
        for term in terms:
            term.parent = self

        return terms

    def expand_spin_orbitals(self):
        """Convert from spin orbitals to spin-labelled spatial orbitals
        by expanding all possible spin configurations and removing
        terms which violate spin orthonormality.
        """
        # TODO move to Tensor class?

        expanded_tensors = []
        for tensor in self.rhs_tensors:
            new_tensor = 0
            for perm, action in tensor.spin_group:
                spin_indices = []
                for index, spin in zip(tensor.indices, perm):
                    spin_indices.append(index.__class__(index.name, index.space, spin=spin))
                new_tensor_contrib = tensor.copy(indices=spin_indices)
                new_tensor_contrib = action_to_callable(action)(new_tensor_contrib)
                new_tensor += new_tensor_contrib
            expanded_tensors.append(new_tensor)

        # Expand the product between expanded_tensors:
        expanded_tensors = expand_products(sympy.Mul(*expanded_tensors))

        # Get the allowed LHS spins:
        if isinstance(self.lhs, Tensor):
            allowed_lhs_spins = set(tuple(perm) for perm, action in self.lhs.spin_group)

        def check(term):
            # Get the indices on each particle in parent term:
            # FIXME parent always the same?
            parent = term
            while parent.parent:
                parent = parent.parent
            index_particles = parent.get_index_particles()
            index_particles = {index.name: particle for index, particle in index_particles.items()}

            ## Check the spin of each index is the same:
            #spins = {}
            #for index in term.indices:
            #    if index in spins:
            #        if spins[index.name] != index.spin: # ???
            #            return False
            #    spins[index.name] = index.spin

            # Check the spin of each index on the same particle is the same:
            spins = {}
            for index in term.indices:
                particle = index_particles[index.name]
                if particle in spins:
                    if spins[particle] != index.spin:
                        return False
                spins[particle] = index.spin

            # Check the spin on the LHS:
            if isinstance(self.lhs, Tensor):
                spins = tuple(index.spin for index in term.lhs.indices)
                if spins not in allowed_lhs_spins:
                    return False

            return True

        terms = []
        for product in expanded_tensors:
            if isinstance(product, Tensor):
                args = (self.factor, product)
            else:
                args = (self.factor, *product.args)

            new_term = Term(self.lhs, args)
            new_term.parent = self

            # Change the LHS indices to spin indices  TODO move?
            if isinstance(new_term.lhs, Tensor):
                inds = []
                for index1 in new_term.lhs.external_indices:
                    for index2 in new_term.rhs_external_indices:
                        if (index1.name, index1.space) == (index2.name, index2.space):
                            inds.append(index2)
                            break
                    else:
                        raise ValueError
                new_term.lhs = new_term.lhs.copy(indices=inds)

            if new_term.factor != 0 and check(new_term):
                terms.append(new_term)

        return terms

    def to_rhf(self, project_onto: List[Tuple[SpinType]] = None):
        """Convert an expression over spatial orbitals with a spin
        tag into a restricted expression.

        project_onto:
            A list of tuples of spins indicating configurations to
            project onto for the LHS. If None, project onto all
            valid spin configurations.
        """

        if project_onto is not None:
            assert all(isinstance(x, tuple) for x in project_onto)
            spins = tuple(index.spin for index in self.lhs.indices)
            if spins not in project_onto:
                return Term(self.lhs, (0,))

        new_rhs = []
        for tensor in self.rhs_tensors:
            restricted_indices = []
            for index in tensor.indices:
                restricted_indices.append(index.__class__(index.name, index.space, spin=None))
            new_tensor = tensor.copy(indices=restricted_indices)
            new_rhs.append(new_tensor)

        new_rhs = [self.factor] + new_rhs

        if isinstance(self.lhs, Tensor):
            restricted_indices = []
            for index in self.lhs.indices:
                restricted_indices.append(index.__class__(index.name, index.space, spin=None))
            new_lhs = self.lhs.copy(indices=restricted_indices)
        else:
            new_lhs = self.lhs

        new_term = Term(new_lhs, new_rhs)
        new_term.parent = self

        return new_term

    def to_uhf(self):
        """Convert an expression over spatial orbitals with a spin
        tag into an unrestricted expression.
        """

        new_rhs = []
        for tensor in self.rhs_tensors:
            restricted_indices = []
            for index in tensor.indices:
                restricted_indices.append(index.__class__(index.name, index.space, spin=None))
            spin = "".join([{ALPHA: "a", BETA: "b"}[index.spin] for index in tensor.indices])
            base = tensor.base.copy(name=tensor.base.name + "_" + spin)
            new_tensor = tensor.copy(base=base, indices=restricted_indices)
            new_rhs.append(new_tensor)

        if isinstance(self.lhs, Tensor):
            restricted_indices = []
            for index in self.lhs.indices:
                restricted_indices.append(index.__class__(index.name, index.space, spin=None))
            spin = "".join([{ALPHA: "a", BETA: "b"}[index.spin] for index in self.lhs.indices])
            base = self.lhs.base.copy(name=self.lhs.base.name + "_" + spin)
            new_lhs = self.lhs.copy(base=base, indices=restricted_indices)
        else:
            new_lhs = self.lhs

        new_term = Term(new_lhs, new_rhs)
        new_term.parent = self

        return new_term


def build_indices_dict(dummies, externals):
    """For dictionaries mapping spaces to indices, add the respective
    spin indices.
    """

    indices = {
            "dummies": {},
            "externals": {},
    }

    for space, inds in dummies.items():
        indices["dummies"][(space, None)] = inds
        indices["dummies"][(space, ALPHA)] = []
        indices["dummies"][(space, BETA)] = []
        for index in inds:
            cls = index.__class__
            indices["dummies"][(space, ALPHA)].append(cls(index.name, index.space, spin=ALPHA))
            indices["dummies"][(space, BETA)].append(cls(index.name, index.space, spin=BETA))

    for space, inds in externals.items():
        indices["externals"][(space, None)] = inds
        indices["externals"][(space, ALPHA)] = []
        indices["externals"][(space, BETA)] = []
        for index in inds:
            cls = index.__class__
            indices["externals"][(space, ALPHA)].append(cls(index.name, index.space, spin=ALPHA))
            indices["externals"][(space, BETA)].append(cls(index.name, index.space, spin=BETA))

    return indices


def combine_terms(terms):
    """Combine terms which are equal within their factors.
    """

    combine_lists = {}

    for term in terms:
        h = hash(term)
        if h in combine_lists:
            combine_lists[h].append(term)
        else:
            combine_lists[h] = [term]

    new_terms = []
    for terms in combine_lists.values():
        new_term = terms[0]
        for term in terms[1:]:
            new_term = new_term + term
        if new_term.factor != 0:
            new_terms.append(new_term)

    return new_terms


def ghf_to_rhf(terms, indices, project_onto: List[Tuple[SpinType]] = None):
    """Convert a list of Terms in a spin-orbital basis to a list of
    Terms over restricted spatial orbitals.
    """

    def flatten(terms):
        out = []
        for term in terms:
            if isinstance(term, (Sequence, sympy.Tuple)):
                out += flatten(term)
            else:
                out.append(term)
        return out

    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    terms = flatten([term.expand_particle_exchange_symmetry() for term in terms])
    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    terms = flatten([term.expand_spin_orbitals() for term in terms])
    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    terms = flatten([term.to_rhf(project_onto=project_onto) for term in terms])
    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    return terms


def ghf_to_uhf(terms, indices):
    """Convert a list of Terms in a spin-orbital basis to a list of
    Terms over unrestricted spatial orbitals.
    """

    raise NotImplementedError  # FIXME

    def flatten(terms):
        out = []
        for term in terms:
            if isinstance(term, (Sequence, sympy.Tuple)):
                out += flatten(term)
            else:
                out.append(term)
        return out

    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    terms = flatten([term.expand_particle_exchange_symmetry() for term in terms])
    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    terms = flatten([term.expand_spin_orbitals() for term in terms])
    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    terms = flatten([term.to_uhf() for term in terms])
    terms = [term.canonicalize(indices=indices) for term in terms]
    terms = combine_terms(terms)

    return terms



if __name__ == "__main__":
    i, j = ExternalIndex("i:j", OCCUPIED)
    k, l, m = DummyIndex("k:m", OCCUPIED)
    a, b = ExternalIndex("a:b", VIRTUAL)
    c, d, e = DummyIndex("c:e", VIRTUAL)

    dummies = {
            OCCUPIED: [k, l, m],
            VIRTUAL: [c, d, e],
    }
    externals = {
            OCCUPIED: [i, j],
            VIRTUAL: [a, b],
    }

    indices = build_indices_dict(dummies, externals)

    v = TensorSymbol("v", 4,
            group=[((0, 1, 2, 3), IDENTITY), ((1, 0, 3, 2), IDENTITY), ((1, 0, 2, 3), NEGATIVE), ((0, 1, 3, 2), NEGATIVE)],
            particles=[(FERMION, 0), (FERMION, 1), (FERMION, 0), (FERMION, 1)],
    )

    res = TensorSymbol("res", 4,
            # How is this being applied automatically...?
            group=[((0, 1, 2, 3), IDENTITY), ((1, 0, 3, 2), IDENTITY), ((1, 0, 2, 3), NEGATIVE), ((0, 1, 3, 2), NEGATIVE)],
            particles=[(FERMION, 0), (FERMION, 1), (FERMION, 0), (FERMION, 1)],
    )

    terms = [
        Term(res[i, j, a, b], [0.25, v[i, j, c, d], v[c, d, a, b]]),
    ]

    #res = sympy.Symbol("res")
    #terms = [
    #    Term(res, [0.25, v[k, l, c, d], v[c, d, k, l]]),
    #]

    for term in terms:
        print(term)
    print()

    #terms = [term.canonicalize(indices=indices) for term in terms]
    #terms = combine_terms(terms)
    #for term in terms:
    #    print(term)
    #print()

    #terms = sum([term.expand_particle_exchange_symmetry() for term in terms], [])
    #terms = [term.canonicalize(indices=indices) for term in terms]
    #terms = combine_terms(terms)
    #for term in terms:
    #    print(term)
    #print()

    #terms = sum([term.expand_spin_orbitals() for term in terms], [])
    #terms = [term.canonicalize(indices=indices) for term in terms]
    #terms = combine_terms(terms)
    #for term in terms:
    #    print(term)
    #print()

    #terms = [term.to_rhf() for term in terms]
    #terms = [term.canonicalize(indices=indices) for term in terms]
    #terms = combine_terms(terms)
    #for term in terms:
    #    print(term)
    #print()

    for term in ghf_to_rhf(terms, indices):
        print(term)

    from qwick.codegen import sympy_to_drudge
    print()
    print(sympy_to_drudge(ghf_to_rhf(terms, indices), indices))
