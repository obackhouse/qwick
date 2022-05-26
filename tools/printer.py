"""Print expressions.
"""

import typing, types
import sympy
import drudge
import gristmill
from sympy.printing.python import PythonPrinter

# TODO add spins for UHF
# TODO change convention for bosonic DOF so we can use capitals for UHF


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
        **{k: "o" for k in (["i", "j", "k", "l", "m", "n"] + ["o%d" % n for n in range(25)])},
        **{k: "v" for k in (["a", "b", "c", "d", "e", "f"] + ["v%d" % n for n in range(25)])},
        **{k: "O" for k in (["I", "J", "K", "L", "M", "N"] + ["O%d" % n for n in range(25)])},
        **{k: "V" for k in (["A", "B", "C", "D", "E", "F"] + ["V%d" % n for n in range(25)])},
}


TEMPLATE = """
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
            **kwargs,
    ):
        super().__init__(add_templ={"einsum_custom": TEMPLATE}, **kwargs)
        self.occupancy_tags = occupancy_tags
        self.reorder_axes = reorder_axes
        self.remove_spacing = remove_spacing

    def proc_ctx(
            self,
            tensor_def: drudge.TensorDef,
            term: typing.Optional[drudge.Term],
            tensor_entry: types.SimpleNamespace,
            term_entry: typing.Optional[types.SimpleNamespace],
    ):
        if term is None:
            self._indexed_proc(tensor_entry, self._print_scal)
        else:
            # FIXME this is a hacky workarund - reconfigure scalar printer
            if term_entry.numerator.startswith("Float"):
                term_entry.numerator = term_entry.numerator.split("\'")[1]
            if term_entry.denominator.startswith("Float"):
                term_entry.denominator = term_entry.denominator.split("\'")[1]

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

    def render(self, templ_name, ctx):
        templ = self._env.get_template("einsum_custom")
        return templ.render(ctx.__dict__)

    def doprint(self, *args, **kwargs):
        string = super().doprint(*args, **kwargs)
        if self.remove_spacing:
            string = string.split("\n")
            string = [s for s in string if s]
            string = "\n".join(string)
        return string
