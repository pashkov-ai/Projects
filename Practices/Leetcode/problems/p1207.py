"""
1207. Unique Number of Occurrences
Easy

Given an array of integers arr, write a function that returns true if and only if the number of occurrences of each value in the array is unique.



Example 1:

Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.

Example 2:

Input: arr = [1,2]
Output: false

Example 3:

Input: arr = [-3,0,1,-3,1,1,1,-3,10,0]
Output: true



Constraints:

    1 <= arr.length <= 1000 // N
    -1000 <= arr[i] <= 1000 // M
"""
from typing import List


# time: O(N+M), space: O(M)
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        value_freq = dict()  # O(M) - space
        for el in arr:  # O(N) - time
            value_freq[el] = value_freq.get(el, 0) + 1

        freqs = set()  # O(M) - space
        for _, f in value_freq.items():  # O(M) - time
            if f in freqs:
                return False
            freqs.add(f)
        return True

    #   freqs = set(value_freq.values())
    #   return len(freqs) == len(value_freq)
