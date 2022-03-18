qwick
=====

C++ version of the [`wick`](https://github.com/awhite862/wick) program for symbolic manipulation of operators in quantum chemistry.

The code is bound to a `pybind11` interface such that it should operate indentically, with only changes required on the import level, for example:

```
from wick.index import Idx
```

should be changed to

```
from qwick.index import Idx
```

For documentation, please see the `wick` repository.
