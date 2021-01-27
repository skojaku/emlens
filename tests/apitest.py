import unittest
import emlens

class TestCalc(unittest.TestCase):
 
  def test_add_num(self):
    self.assertEqual(10, calc.add_num(6, 4)) 
 
  def test_sub_num(self):
    self.assertEqual(2, calc.sub_num(6, 4))

  def test_mul_num(self):
    self.assertEqual(24, calc.mul_num(6, 4))

  def test_div_num(self):
    self.assertEqual(10, calc.div_num(6, 4))
