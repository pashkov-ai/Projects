"""
9. Palindrome Number
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

Example 1:

Input: 121
Output: true

Example 2:

Input: -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.

Example 3:

Input: 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.

Follow up:

Coud you solve it without converting the integer to a string?
"""

from collections import deque


class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        deq = deque()
        while x > 0:
            deq.append(x % 10)
            x = x // 10
        while True:
            try:
                if deq.pop() != deq.popleft():
                    return False
            except IndexError:
                return True
