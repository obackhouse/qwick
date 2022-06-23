"""Printing functions.
"""

import typing, types
import sympy
import drudge
import gristmill
from sympy.printing.python import PythonPrinter

# TODO conversion of wick/sympy expressions
# TODO add spins for UHF
# TODO change convention for bosonic DOF so we can use capitals for UHF
# FIXME how is this going to convert i.e. o1 to a char?


def format_float(x, places=14):
    if isinstance(x, complex):
        raise NotImplementedError
    sx = "%.*f" % (places, x)
    while sx[-1] == "0" and sx[-2] != ".":
        sx = sx[:-1]
    if sx == "1.0":
        return None
    return sx


index_to_sector = {
        **{k: "o" for k in (["i", "j", "k", "l", "m", "n", "o", "p", "I", "J", "K", "L", "M", "N", "O", "P"])},
        **{k: "v" for k in (["a", "b", "c", "d", "e", "f", "g", "h", "A", "B", "C", "D", "E", "F", "G", "H"])},
        **{k: "b" for k in (["u", "v", "w", "x", "y", "z", "U", "V", "W", "X", "Y", "Z"])},
}


EINSUM_TEMPLATE = """
{{ base }} {{ term.phase }}= {{ einsum }}("{% for factor in term.indexed_factors %}
{% for i in factor.indices %}{{ i.index }}{% endfor %}
{% if not loop.last %},{% endif %}
{% endfor %}->{% for i in indices %}{{ i.index }}{% endfor %}", {% for factor in term.indexed_factors %}
{{ factor.base }}{% if not loop.last %}, {% endif %}{% endfor %}){% if term.numerator != '1' %}
 * {{ term.numerator }}{% endif %}{% if term.denominator != '1' -%}
{{ ' /' }} {{ term.denominator }}{% endif %}
"""


class EinsumPrinter(gristmill.EinsumPrinter):
    _scal_printer = PythonPrinter

    def __init__(
            self,
            occupancy_tags={},
            reorder_axes={},
            remove_spacing=True,
            explicit_init=True,
            garbage_collection=True,
            **kwargs,
    ):
        super().__init__(add_templ={"einsum_custom": EINSUM_TEMPLATE}, **kwargs)

        self.occupancy_tags = occupancy_tags
        self.reorder_axes = reorder_axes
        self.remove_spacing = remove_spacing
        self.explicit_init = explicit_init
        self.garbage_collection = garbage_collection

        if not self.explicit_init:
            raise NotImplementedError(
                    "`explicit_init = False` may result in references "
                    "to arrays being mutated. Need to implement this "
                    "better."
            )

    def proc_ctx(
            self,
            tensor_def: drudge.TensorDef,
            term: typing.Optional[drudge.Term],
            tensor_entry: types.SimpleNamespace,
            term_entry: typing.Optional[types.SimpleNamespace],
    ):
        if term is None:
            # If we do not explicitly initialise tensors, make sure first
            # term is an assignment and not increment/decrement
            if not self.explicit_init:
                if tensor_entry.terms[0].phase == "+":
                    tensor_entry.terms[0].phase = ""
                elif tensor_entry.terms[0].phase == "-":
                    tensor_entry.terms[0].phase = ""
                    tensor_entry.terms[0].numerator = str(-float(tensor_entry.terms[0].numerator))
                else:
                    raise Exception("Edge cases?")

            # Reorder axes of desired tensors
            if tensor_entry.base in self.reorder_axes:
                perm = self.reorder_axes[tensor_entry.base]
                tensor_entry.indices = [tensor_entry.indices[p] for p in perm]

            self._indexed_proc(tensor_entry, self._print_scal)

        else:
            for i in term_entry.indexed_factors:
                # Reorder axes of desired tensors
                if i.base in self.reorder_axes:
                    perm = self.reorder_axes[i.base]
                    i.indices = [i.indices[p] for p in perm]

                # Add occupancy tags to desired tensors
                if i.base in self.occupancy_tags:
                    indices = [index.index for index in i.indices]
                    tags = [index_to_sector[index] for index in indices]
                    template = self.occupancy_tags[i.base]
                    i.base = template.format(base=i.base, tags="".join(tags))

                self._indexed_proc(i, self._print_scal)
                continue

        return

    def print_before_comp(self, event):
        if self.explicit_init:
            return super().print_before_comp(event)
        return None

    def print_out_of_use(self, event):
        if self.garbage_collection:
            return super().print_out_of_use(event)
        return None

    def _print_scal(self, expr):
        # FIXME a little bit hacky
        if isinstance(expr, sympy.core.numbers.Float):
            expr = float(expr)
            if abs(expr - int(expr)) < 1e-15:
                expr = int(expr)
        return self._scal_printer.doprint(expr)

    def render(self, templ_name, ctx):
        templ = self._env.get_template("einsum_custom")
        return templ.render(ctx.__dict__)

    def doprint(self, *args, **kwargs):
        string = super().doprint(*args, **kwargs)
        if self.remove_spacing:
            string = string.split("\n")
            string = [s for s in string if s.strip()]
            string = "\n".join(string)
        return string



