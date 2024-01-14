import unittest
from typing import List

class SearchInsert():
    def run(self, nums: List[int], target: int) -> int:
        # binary search solution
        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        return low

class TestCode(unittest.TestCase):
    def setUp(self):
        self.search_insert = SearchInsert()
    
    def test_search_insert(self):
        self.assertEqual(self.search_insert.run([1,3,5,6], 5), 2)
        self.assertEqual(self.search_insert.run([1,3,5,6], 2), 1)
        self.assertEqual(self.search_insert.run([1,3,5,6], 7), 4)
        self.assertEqual(self.search_insert.run([1], 0), 0)
        self.assertEqual(self.search_insert.run([1], 1), 0)
        self.assertEqual(self.search_insert.run([1,3], 2), 1)

if __name__ == "__main__":
    unittest.main()

"""
35. Search Insert Position

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [1,3,5,6], target = 5
Output: 2
Example 2:

Input: nums = [1,3,5,6], target = 2
Output: 1
Example 3:

Input: nums = [1,3,5,6], target = 7
Output: 4
 

Constraints:

1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums contains distinct values sorted in ascending order.
-104 <= target <= 104
"""
