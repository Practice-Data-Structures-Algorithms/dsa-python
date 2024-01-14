import unittest

class RomanToInt:
    def run(self, s : str) -> int:
        l_s = list(s)
        l_s.reverse()
        rom_nums = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }
        res: int = 0
        prev_num: int = 0
        for num in l_s:
            if rom_nums[num] < prev_num:
                res -= rom_nums[num]
            else:
                res += rom_nums[num]
            prev_num = rom_nums[num]
        return res

class TestCode(unittest.TestCase):
    def setUp(self):
        self.roman_to_int = RomanToInt()

    def test_roman_to_int(self):
        self.assertEqual(self.roman_to_int.run("III"), 3)
        self.assertEqual(self.roman_to_int.run("LVIII"), 58)
        self.assertEqual(self.roman_to_int.run("MCMXCIV"), 1994)

if __name__ == "__main__":
    unittest.main()

"""
Input: s = "MCMXCIV"
1000 100 1000 10 100 1 5
5 - 1 + 100 - 10 + 1000 - 100 + 1000
Output: 1994

rules: {
    I: V, X
    X: L, C
    C: D, M
}
"""
        
"""
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].
        """
