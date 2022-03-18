"""Run wick tests using qwick backend.
"""

import unittest

from qwick.tests.test_aterm import *
from qwick.tests.test_convenience import *
from qwick.tests.test_expression import *
from qwick.tests.test_full import *
from qwick.tests.test_idx import *
from qwick.tests.test_operators import *
from qwick.tests.test_sc_rules import *
from qwick.tests.test_term import *
from qwick.tests.test_term_map import *
from qwick.tests.test_test import *
from qwick.tests.test_wick import *

if __name__ == "__main__":
    unittest.main()
