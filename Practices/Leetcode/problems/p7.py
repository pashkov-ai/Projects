"""
7. Reverse Integer
Given a 32-bit signed integer, reverse digits of an integer.

Example 1:

Input: 123
Output: 321

Example 2:

Input: -123
Output: -321

Example 3:

Input: 120
Output: 21

Note:
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.
"""
import math


class Solution:
    def reverse(self, x: int) -> int:
        max_overflow = 2 ** 31 if x < 0 else 2 ** 31 - 1
        revers = 0
        t = abs(x)
        while t > 0:
            if max_overflow / 10 < revers:
                return 0
            revers = 10 * revers
            if max_overflow - (t % 10) < revers:
                return 0
            revers += (t % 10)
            t = t // 10
        return int(math.copysign(1, x) * revers)
