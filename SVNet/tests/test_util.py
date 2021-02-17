import unittest

from lib.utils import *

class TestBits(unittest.TestCase):
    def test_is_aliased(self):
        self.assertTrue(is_aliased(X86.AL, X86.RAX))
        self.assertFalse(is_aliased(X86.AL, X86.BL))

    def test_renumber_reg(self):
        self.assertEqual(renumber_reg(X86.RAX), 0)
        self.assertEqual(renumber_reg(X86.XMM5), 30)
        self.assertEqual(renumber_reg(X86.CR0), -1)
