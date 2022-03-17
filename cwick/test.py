"""Run wick tests using cwick backend.
"""

import unittest

from cwick.tests.test_aterm import *
from cwick.tests.test_convenience import *
from cwick.tests.test_expression import *
from cwick.tests.test_full import *
from cwick.tests.test_idx import *
from cwick.tests.test_operators import *
from cwick.tests.test_sc_rules import *
from cwick.tests.test_term import *
from cwick.tests.test_term_map import *
from cwick.tests.test_test import *
from cwick.tests.test_wick import *

if __name__ == "__main__":
    unittest.main()
