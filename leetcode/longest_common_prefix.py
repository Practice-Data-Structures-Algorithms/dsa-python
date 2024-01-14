from typing import List

class LongestCommonPrefix:
    def run(self, strs: List[str]) -> str:
        strs.sort()
        first: str = strs[0]
        last: str = strs[-1]
        longest: int = 0

        for i in range(len(first)):
            if first[i] == last[i]:
                longest += 1
            else:
                break

        return first[0:longest]

"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters.
"""